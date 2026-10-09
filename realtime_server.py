"""Browser-based realtime audio driving for May and Macron checkpoints.

Audio path:
    browser microphone -> WebSocket 16 kHz PCM -> DeepSpeech worker (GPU 1)
    -> (8, 29, 16) feature queue -> TalkingGaussian renderer (GPU 0)

Video path:
    persistent single-frame renderer -> JPEG -> bounded WebSocket video stream

This is intentionally a single-audio-session research server.
"""

import asyncio
import copy
import json
import os
import queue
import signal
import ssl
import struct
import threading
import time
from argparse import ArgumentParser
from collections import deque

from aiohttp import WSMsgType, web
import numpy as np

from arguments import ModelParams, PipelineParams, get_combined_args
from realtime.deepspeech_stream import StreamingDeepSpeechProcess
from realtime.latency_trace import frame_metadata, unpack_audio_packet


class FrameHub:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.sequence = 0
        self.updated_at = 0.0
        self.trace = None

    def publish(self, frame, trace=None):
        with self.lock:
            self.frame = frame
            self.sequence += 1
            self.updated_at = time.time()
            self.trace = trace

    def snapshot(self):
        with self.lock:
            return self.frame, self.sequence, self.updated_at

    def traced_snapshot(self):
        with self.lock:
            return self.frame, self.sequence, self.updated_at, self.trace


class RealtimeService:
    def __init__(
        self,
        profiles,
        asr,
        fps=25,
        feature_fps=25,
        max_feature_lag=2,
        speech_rms_threshold=0.006,
        speech_rms_full=0.05,
    ):
        self.profiles = profiles
        self.avatar_lock = threading.RLock()
        self.active_avatar = "may"
        self.renderer = profiles[self.active_avatar]["renderer"]
        self.avatar_generation = 0
        self.asr = asr
        self.fps = float(fps)
        self.feature_fps = float(feature_fps)
        if self.fps <= 0 or self.feature_fps <= 0:
            raise ValueError("Frame rates must be positive")
        self.max_feature_lag = int(max_feature_lag)
        self.speech_rms_threshold = float(speech_rms_threshold)
        self.speech_rms_full = float(speech_rms_full)
        self.reference_features = profiles[self.active_avatar]["reference_features"]
        self.reference_queue = queue.Queue(maxsize=4)
        self.reference_samples_seen = 0
        self.reference_frames_emitted = 0
        self.reference_previous = None
        self.reference_mode = False
        if self.speech_rms_full <= self.speech_rms_threshold:
            raise ValueError("speech_rms_full must be greater than speech_rms_threshold")
        self.frame_hub = FrameHub()
        self.stop_event = threading.Event()
        self.render_thread = None
        self.status_lock = threading.Lock()
        self.status = {
            "state": "starting",
            "render_frames": 0,
            "render_fps": 0.0,
            "render_ms": 0.0,
            "dropped_feature_frames": 0,
            "audio_clients": 0,
            "audio_bytes": 0,
            "video_clients": 0,
            "video_ws_clients": 0,
            "video_bytes": 0,
            "audio_drive_active": False,
            "input_source": "none",
            "reference_feature_frames": 0,
        }
        self.last_asr_status = {}
        self.polling_clients = {}
        self.traffic_samples = {
            "uplink": deque(),
            "downlink": deque(),
        }

    def start(self):
        self.render_thread = threading.Thread(
            target=self._render_loop, name="talking-gaussian-render", daemon=True
        )
        self.render_thread.start()

    def stop(self):
        self.stop_event.set()
        if self.render_thread is not None:
            self.render_thread.join(timeout=5.0)
        self.asr.stop()

    def _set_status(self, **values):
        with self.status_lock:
            self.status.update(values)

    def select_avatar(self, avatar):
        if avatar not in self.profiles:
            raise ValueError("Unknown avatar: {}".format(avatar))
        if avatar == self.active_avatar:
            return self.avatar_metadata()
        self.asr.reset()
        while True:
            try:
                self.asr.feature_queue.get_nowait()
            except queue.Empty:
                break
        self.reference_mode = False
        while True:
            try:
                self.reference_queue.get_nowait()
            except queue.Empty:
                break
        with self.avatar_lock:
            self.active_avatar = avatar
            self.renderer = self.profiles[avatar]["renderer"]
            self.reference_features = self.profiles[avatar]["reference_features"]
            self.reference_samples_seen = 0
            self.reference_frames_emitted = 0
            self.reference_previous = None
            self.renderer.reset_motion()
            self.avatar_generation += 1
        self._set_status(input_source="none", reference_feature_frames=0)
        return self.avatar_metadata()

    def avatar_metadata(self):
        with self.avatar_lock:
            profile = self.profiles[self.active_avatar]
            return {
                "avatar": self.active_avatar,
                "label": profile["label"],
                "audio_start": profile["audio_start"],
                "audio_duration": profile["audio_duration"],
                "validation_frames": profile["validation_frames"],
            }

    def record_traffic(self, direction, byte_count):
        if direction not in self.traffic_samples:
            raise ValueError("Unknown traffic direction: {}".format(direction))
        byte_count = int(byte_count)
        if byte_count <= 0:
            return
        now = time.monotonic()
        status_key = "audio_bytes" if direction == "uplink" else "video_bytes"
        with self.status_lock:
            self.status[status_key] += byte_count
            samples = self.traffic_samples[direction]
            samples.append((now, byte_count))
            cutoff = now - 3.0
            while samples and samples[0][0] < cutoff:
                samples.popleft()

    def start_reference(self):
        if self.reference_features is None:
            raise RuntimeError("Reference features are unavailable")
        while True:
            try:
                self.reference_queue.get_nowait()
            except queue.Empty:
                break
        self.reference_samples_seen = 0
        self.reference_frames_emitted = 0
        self.reference_previous = None
        self.reference_mode = True
        self.renderer.reset_motion()
        self._set_status(reference_feature_frames=0)

    def push_reference_pcm(self, pcm_bytes):
        if len(pcm_bytes) % 2:
            raise ValueError("PCM payload must contain complete int16 samples")
        self.reference_samples_seen += len(pcm_bytes) // 2
        samples_per_frame = int(round(self.asr.sample_rate / self.feature_fps))
        target_frames = min(
            len(self.reference_features),
            self.reference_samples_seen // samples_per_frame,
        )
        pcm = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32)
        rms = float(np.sqrt(np.mean(pcm * pcm) + 1e-12) / 32768.0)
        while self.reference_frames_emitted < target_frames:
            index = self.reference_frames_emitted
            features = self.reference_features[index]
            delta = 0.0 if self.reference_previous is None else float(
                np.mean(np.abs(features - self.reference_previous))
            )
            item = {
                "features": features,
                "rms": rms,
                "feature_delta": delta,
                "feature_std": float(np.std(features)),
                "audio_seconds": (index + 1) / self.feature_fps,
                "generated_at": time.monotonic(),
            }
            if self.reference_queue.full():
                try:
                    self.reference_queue.get_nowait()
                    with self.status_lock:
                        self.status["dropped_feature_frames"] += 1
                except queue.Empty:
                    pass
            self.reference_queue.put_nowait(item)
            self.reference_previous = features
            self.reference_frames_emitted += 1
        self._set_status(reference_feature_frames=self.reference_frames_emitted)

    def _traffic_rate_locked(self, direction, now):
        samples = self.traffic_samples[direction]
        cutoff = now - 2.0
        while samples and samples[0][0] < cutoff:
            samples.popleft()
        if not samples:
            return 0.0
        duration = max(1.0, min(2.0, now - samples[0][0]))
        return sum(item[1] for item in samples) / duration

    def _drain_asr_status(self):
        while True:
            try:
                self.last_asr_status = self.asr.status_queue.get_nowait()
            except queue.Empty:
                break

    def _trim_feature_backlog(self):
        dropped = 0
        try:
            target_queue = (
                self.reference_queue if self.reference_mode else self.asr.feature_queue
            )
            while target_queue.qsize() > self.max_feature_lag:
                target_queue.get_nowait()
                dropped += 1
        except (NotImplementedError, queue.Empty):
            pass
        if dropped:
            with self.status_lock:
                self.status["dropped_feature_frames"] += dropped

    def _render_loop(self):
        period = 1.0 / self.fps
        next_tick = time.perf_counter()
        current_features = self.renderer.neutral_audio
        current_rms = 0.0
        current_feature_delta = 0.0
        current_feature_std = 0.0
        current_feature_audio_seconds = 0.0
        current_feature_generated_at = 0.0
        current_trace = None
        natural_motion_strength = 0.0
        last_feature_at = 0.0
        frame_times = []
        wall_times = []
        publish_times = []
        self._set_status(state="warming_renderer")
        warmup = self.renderer.warm_up(3)
        self._set_status(
            state="ready",
            warmup_ms=[round(value, 2) for value in warmup],
        )

        while not self.stop_event.is_set():
            with self.avatar_lock:
                generation = self.avatar_generation
                renderer = self.renderer
            if getattr(self, "_render_generation", -1) != generation:
                self._render_generation = generation
                current_features = renderer.neutral_audio
                current_rms = current_feature_delta = current_feature_std = 0.0
                current_feature_audio_seconds = current_feature_generated_at = 0.0
                natural_motion_strength = last_feature_at = 0.0
                current_trace = None
            self._trim_feature_backlog()
            audio_drive_active = bool(
                last_feature_at and time.monotonic() - last_feature_at <= 0.6
            )
            try:
                target_queue = (
                    self.reference_queue if self.reference_mode else self.asr.feature_queue
                )
                feature_item = target_queue.get_nowait()
                current_trace = feature_item.get("trace") if isinstance(feature_item, dict) else None
                if isinstance(feature_item, dict):
                    current_features = feature_item["features"]
                    current_rms = float(feature_item.get("rms", 0.0))
                    current_feature_delta = float(
                        feature_item.get("feature_delta", 0.0)
                    )
                    current_feature_std = float(
                        feature_item.get("feature_std", 0.0)
                    )
                    current_feature_audio_seconds = float(
                        feature_item.get("audio_seconds", 0.0)
                    )
                    current_feature_generated_at = float(
                        feature_item.get("generated_at", 0.0)
                    )
                else:
                    current_features = feature_item
                last_feature_at = time.monotonic()
                audio_drive_active = True
            except queue.Empty:
                if last_feature_at and time.monotonic() - last_feature_at > 0.6:
                    current_features = self.renderer.neutral_audio
                    current_rms = 0.0
                    current_feature_delta = 0.0
                    current_feature_std = 0.0
                    current_feature_generated_at = 0.0
                    audio_drive_active = False
                    current_trace = None

            if audio_drive_active:
                target_motion = (current_rms - self.speech_rms_threshold) / (
                    self.speech_rms_full - self.speech_rms_threshold
                )
                target_motion = max(0.0, min(1.0, target_motion))
            else:
                target_motion = 0.0
            smoothing = 0.35 if target_motion > natural_motion_strength else 0.16
            natural_motion_strength += smoothing * (
                target_motion - natural_motion_strength
            )
            if natural_motion_strength < 1e-4:
                natural_motion_strength = 0.0

            wall_started = time.perf_counter()
            render_started = time.monotonic()
            try:
                jpeg, render_ms = renderer.render_jpeg(
                    current_features,
                    motion_strength=natural_motion_strength,
                )
                if generation != self.avatar_generation:
                    continue
                trace = dict(current_trace, render_started=render_started,
                             render_finished=time.monotonic(), avatar=self.active_avatar,
                             avatar_generation=generation) if current_trace else None
                self.frame_hub.publish(jpeg, trace)
            except Exception as error:
                self._set_status(state="render_error", error=repr(error))
                time.sleep(0.1)
                continue

            wall_ms = (time.perf_counter() - wall_started) * 1000.0
            frame_times.append(render_ms)
            wall_times.append(wall_ms)
            publish_times.append(time.perf_counter())
            frame_times = frame_times[-100:]
            wall_times = wall_times[-100:]
            publish_times = publish_times[-100:]
            stream_fps = 0.0
            if len(publish_times) > 1:
                stream_fps = (len(publish_times) - 1) / (
                    publish_times[-1] - publish_times[0]
                )
            self._drain_asr_status()
            self._set_status(
                state="ready",
                render_frames=self.frame_hub.sequence,
                render_ms=round(sum(frame_times) / len(frame_times), 2),
                render_fps=round(stream_fps, 2),
                audio_drive_active=audio_drive_active,
                speech_active=natural_motion_strength > 0.05,
                audio_rms=round(current_rms, 6),
                natural_motion_strength=round(natural_motion_strength, 4),
                feature_delta=round(current_feature_delta, 5),
                feature_std=round(current_feature_std, 5),
                feature_audio_seconds=round(current_feature_audio_seconds, 3),
                feature_queue_age_ms=round(
                    max(0.0, (time.monotonic() - current_feature_generated_at) * 1000.0),
                    1,
                ) if current_feature_generated_at else 0.0,
                render_capacity_fps=round(
                    1000.0 / max(1e-6, sum(wall_times) / len(wall_times)), 2
                ),
            )

            next_tick += period
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -period * 3:
                next_tick = time.perf_counter()

    def public_status(self):
        self._drain_asr_status()
        now = time.monotonic()
        with self.status_lock:
            expired = [key for key, last_seen in self.polling_clients.items()
                       if now - last_seen > 3.0]
            for key in expired:
                del self.polling_clients[key]
            result = dict(self.status)
            result["video_clients"] += len(self.polling_clients)
            result["uplink_bytes_per_second"] = round(
                self._traffic_rate_locked("uplink", now), 1
            )
            result["downlink_bytes_per_second"] = round(
                self._traffic_rate_locked("downlink", now), 1
            )
        result["asr_alive"] = self.asr.is_alive()
        result["asr"] = dict(self.last_asr_status)
        result["camera"] = self.renderer.get_control()
        result.update(self.avatar_metadata())
        result["avatars"] = list(self.profiles)
        result["target_fps"] = self.fps
        result["audio_feature_fps"] = self.feature_fps
        result["server_time_ms"] = round(time.time() * 1000.0, 3)
        result["speech_rms_threshold"] = self.speech_rms_threshold
        result["feature_mode"] = (
            "reference_features" if self.reference_mode else "rolling_deepspeech"
        )
        result["algorithmic_audio_context_ms"] = int(
            (8 + self.asr.lookahead_logits + 6) * 20
        )
        try:
            result["feature_queue"] = (
                self.reference_queue.qsize()
                if self.reference_mode else self.asr.feature_queue.qsize()
            )
            result["audio_queue"] = self.asr.audio_queue.qsize()
        except NotImplementedError:
            pass
        return result


def create_app(service, html_path, ca_certificate_path=None):
    routes = web.RouteTableDef()

    @routes.get("/")
    async def index(_request):
        return web.FileResponse(html_path, headers={"Cache-Control": "no-store"})

    @routes.get("/latency_metrics.js")
    async def latency_metrics(_request):
        return web.FileResponse(os.path.join(os.path.dirname(html_path), "latency_metrics.js"),
                                headers={"Cache-Control": "no-store"})

    @routes.get("/campus-ca.crt")
    async def campus_ca(_request):
        if not ca_certificate_path or not os.path.isfile(ca_certificate_path):
            raise web.HTTPNotFound()
        return web.FileResponse(
            ca_certificate_path,
            headers={
                "Content-Disposition": (
                    'attachment; filename="TalkingGaussian-Campus-CA.crt"'
                ),
                "Cache-Control": "no-store",
            },
        )

    @routes.get("/media/{avatar}.wav")
    async def validation_audio(request):
        profile = service.profiles.get(request.match_info["avatar"])
        if profile is None:
            raise web.HTTPNotFound()
        return web.FileResponse(profile["audio_path"])

    @routes.get("/media/{avatar}/offline.mp4")
    async def offline_validation(request):
        profile = service.profiles.get(request.match_info["avatar"])
        if profile is None or not os.path.isfile(profile["offline_video_path"]):
            raise web.HTTPNotFound()
        return web.FileResponse(profile["offline_video_path"])

    @routes.get("/frame.jpg")
    async def latest_frame(request):
        frame, sequence, produced_at, trace = service.frame_hub.traced_snapshot()
        if frame is None:
            return web.Response(status=503, text="Renderer warming up")
        client_id = request.query.get("client", "")[:80]
        if client_id:
            with service.status_lock:
                service.polling_clients[client_id] = time.monotonic()
        service.record_traffic("downlink", len(frame))
        return web.Response(
            body=frame,
            content_type="image/jpeg",
            headers={
                "Cache-Control": "no-store, max-age=0",
                "X-Frame-Sequence": str(sequence),
                "X-Frame-Produced-At": "{:.6f}".format(produced_at),
                "X-Frame-Trace": json.dumps(frame_metadata(trace, time.monotonic())),
            },
        )

    @routes.get("/ws/video")
    async def video_socket(request):
        # A four-frame credit window pipelines frames across a high-RTT SSH
        # tunnel while bounding how much stale video can be in flight.
        websocket = web.WebSocketResponse(heartbeat=20.0, compress=False)
        await websocket.prepare(request)
        client_id = request.query.get("client", "")[:80]
        with service.status_lock:
            if client_id:
                service.polling_clients.pop(client_id, None)
            service.status["video_clients"] += 1
            service.status["video_ws_clients"] += 1
        window_size = 0
        outstanding = 0
        paused = False
        last_sequence = -1
        try:
            while not service.stop_event.is_set() and not websocket.closed:
                try:
                    message = await websocket.receive(timeout=0.005)
                except asyncio.TimeoutError:
                    message = None
                if message is not None:
                    if message.type == WSMsgType.TEXT:
                        try:
                            command = json.loads(message.data)
                        except ValueError:
                            command = {}
                        if command.get("command") in ("ready", "window"):
                            try:
                                requested_window = int(command.get("credits", 4))
                            except (TypeError, ValueError):
                                requested_window = 4
                            window_size = max(1, min(4, requested_window))
                            if command.get("command") == "ready":
                                paused = bool(command.get("paused", False))
                        elif command.get("command") == "ack":
                            outstanding = max(0, outstanding - 1)
                        elif command.get("command") == "pause":
                            paused = True
                        elif command.get("command") == "resume":
                            paused = False
                    elif message.type in (WSMsgType.CLOSE, WSMsgType.CLOSING,
                                          WSMsgType.CLOSED, WSMsgType.ERROR):
                        break
                if paused or window_size <= 0 or outstanding >= window_size:
                    continue
                frame, sequence, produced_at, trace = service.frame_hub.traced_snapshot()
                if frame is None or sequence == last_sequence:
                    continue
                packet = struct.pack("!Id", sequence & 0xffffffff, produced_at)
                if request.query.get("trace") == "1":
                    metadata = json.dumps(frame_metadata(trace, time.monotonic()),
                                          separators=(",", ":")).encode("utf-8")
                    packet += struct.pack("!I", len(metadata)) + metadata
                packet += frame
                await websocket.send_bytes(packet)
                service.record_traffic("downlink", len(packet))
                last_sequence = sequence
                outstanding += 1
        except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
            pass
        finally:
            with service.status_lock:
                service.status["video_clients"] = max(0, service.status["video_clients"] - 1)
                service.status["video_ws_clients"] = max(0, service.status["video_ws_clients"] - 1)
            await websocket.close()
        return websocket

    @routes.get("/api/status")
    async def status(_request):
        return web.json_response(service.public_status())

    @routes.post("/api/avatar")
    async def change_avatar(request):
        payload = await request.json()
        avatar = payload.get("avatar")
        if avatar not in service.profiles:
            raise web.HTTPBadRequest(text="Unknown avatar")
        return web.json_response(service.select_avatar(avatar))

    @routes.post("/api/control")
    async def control(request):
        payload = await request.json()
        values = service.renderer.set_control(
            yaw=payload.get("yaw"),
            pitch=payload.get("pitch"),
            roll=payload.get("roll"),
            zoom=payload.get("zoom"),
            expression_intensity=payload.get("expression_intensity"),
        )
        return web.json_response(values)

    @routes.get("/video.mjpg")
    async def video_stream(request):
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "multipart/x-mixed-replace; boundary=frame",
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "Pragma": "no-cache",
            },
        )
        await response.prepare(request)
        with service.status_lock:
            service.status["video_clients"] += 1
        last_sequence = -1
        try:
            while not service.stop_event.is_set():
                frame, sequence, _ = service.frame_hub.snapshot()
                if frame is None or sequence == last_sequence:
                    await asyncio.sleep(0.01)
                    continue
                last_sequence = sequence
                packet = (
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(frame)).encode("ascii")
                    + b"\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
                await response.write(packet)
                service.record_traffic("downlink", len(packet))
        except (BrokenPipeError, ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            with service.status_lock:
                service.status["video_clients"] = max(
                    0, service.status["video_clients"] - 1
                )
        return response

    @routes.get("/ws/audio")
    async def audio_socket(request):
        websocket = web.WebSocketResponse(heartbeat=20.0, max_msg_size=2 ** 20)
        await websocket.prepare(request)
        service.asr.reset()
        with service.status_lock:
            service.status["audio_clients"] += 1
            service.status["input_source"] = "websocket"
        session_source = "websocket"
        try:
            async for message in websocket:
                if message.type == WSMsgType.BINARY:
                    received_at = time.monotonic()
                    pcm, trace = message.data, None
                    if request.query.get("trace") == "1":
                        try:
                            pcm, trace = unpack_audio_packet(message.data, received_at)
                        except (ValueError, TypeError, UnicodeError):
                            await websocket.close(code=1003, message=b"Invalid trace packet")
                            break
                    if session_source in ("reference_features", "may_reference"):
                        service.push_reference_pcm(pcm)
                    else:
                        service.asr.push_pcm(pcm, trace)
                    service.record_traffic("uplink", len(message.data))
                elif message.type == WSMsgType.TEXT:
                    try:
                        payload = json.loads(message.data)
                    except ValueError:
                        payload = {}
                    if payload.get("command") == "reset":
                        service.asr.reset()
                        source = payload.get("source")
                        if source in ("microphone", "dataset_audio", "reference_features", "may", "may_reference"):
                            session_source = source
                            service.reference_mode = False
                            if source in ("reference_features", "may_reference"):
                                service.start_reference()
                            with service.status_lock:
                                service.status["input_source"] = source
                        if source in ("microphone", "dataset_audio", "may"):
                            service.renderer.reset_motion()
                elif message.type == WSMsgType.ERROR:
                    break
        finally:
            if session_source in ("reference_features", "may_reference"):
                service.reference_mode = False
            with service.status_lock:
                service.status["audio_clients"] = max(
                    0, service.status["audio_clients"] - 1
                )
                if service.status["audio_clients"] == 0:
                    service.status["input_source"] = "none"
        return websocket

    app = web.Application(client_max_size=2 ** 20)
    app.add_routes(routes)

    async def cleanup(_app):
        service.stop()

    app.on_cleanup.append(cleanup)
    return app


async def run_sites(app, host, http_port, https_port=0, ssl_context=None):
    """Run the same stateful demo on HTTP and optional HTTPS listeners."""
    runner = web.AppRunner(app)
    await runner.setup()
    loop = asyncio.get_running_loop()
    stop_requested = asyncio.Event()
    installed_signals = []
    try:
        http_site = web.TCPSite(runner, host, http_port)
        await http_site.start()
        print("[REALTIME_URL] http://{}:{}".format(host, http_port))
        if https_port:
            if ssl_context is None:
                raise RuntimeError("HTTPS port requires an SSL context")
            https_site = web.TCPSite(
                runner, host, https_port, ssl_context=ssl_context
            )
            await https_site.start()
            print("[REALTIME_HTTPS_URL] https://{}:{}".format(host, https_port))
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop_requested.set)
                installed_signals.append(sig)
            except NotImplementedError:
                pass
        await stop_requested.wait()
    finally:
        for sig in installed_signals:
            loop.remove_signal_handler(sig)
        await runner.cleanup()


def load_profile(label, dataset, pipeline, args):
    from realtime.avatar_renderer import RealtimeAvatarRenderer

    renderer = RealtimeAvatarRenderer(
        dataset=dataset,
        pipeline=pipeline,
        render_gpu=args.render_gpu,
        camera_index=args.camera_index,
        jpeg_quality=args.jpeg_quality,
        yaw=args.yaw,
        pitch=args.pitch,
        roll=args.roll,
        zoom=args.zoom,
    )
    validation_path = os.path.join(dataset.source_path, "transforms_val.json")
    with open(validation_path, "r") as validation_file:
        validation_frames = json.load(validation_file).get("frames", [])
    if not validation_frames:
        raise RuntimeError("{} requires validation frames".format(label))
    feature_fps = 25.0
    audio_start = float(validation_frames[0]["img_id"]) / feature_fps
    audio_duration = float(len(validation_frames)) / feature_fps
    audio_path = os.path.join(dataset.source_path, "aud.wav")
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(audio_path)
    offline_features = np.load(
        os.path.join(dataset.source_path, "aud_ds.npy"), mmap_mode="r"
    )
    reference_features = []
    for frame in validation_frames:
        frame_id = int(frame["img_id"])
        context = np.zeros((8, 16, 29), dtype=np.float32)
        left = max(0, frame_id - 4)
        right = min(len(offline_features), frame_id + 4)
        context[left - (frame_id - 4):right - (frame_id - 4)] = (
            offline_features[left:right]
        )
        reference_features.append(context.transpose(0, 2, 1).astype(np.float32))
    return {
        "label": label,
        "renderer": renderer,
        "audio_path": audio_path,
        "audio_start": audio_start,
        "audio_duration": audio_duration,
        "validation_frames": len(validation_frames),
        "reference_features": np.stack(reference_features),
        "offline_video_path": os.path.join(
            dataset.model_path, "test", "ours_None", "renders", "out_with_audio.mp4"
        ),
    }


def main():
    parser = ArgumentParser(description="TalkingGaussian realtime browser server")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--https_port", type=int, default=0)
    parser.add_argument("--ssl_cert", type=str, default="")
    parser.add_argument("--ssl_key", type=str, default="")
    parser.add_argument("--ssl_ca", type=str, default="")
    parser.add_argument("--render_gpu", type=int, default=0)
    parser.add_argument("--asr_gpu", type=str, default="1")
    parser.add_argument("--macron_source", type=str, default="data/macron")
    parser.add_argument("--macron_model", type=str, default="output/macron")
    parser.add_argument("--fps", type=int, default=25,
                        help="Output render FPS; audio features remain at 25 FPS")
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--jpeg_quality", type=int, default=88)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--roll", type=float, default=0.0)
    parser.add_argument("--zoom", type=float, default=1.0)
    parser.add_argument("--asr_block_ms", type=int, default=80)
    parser.add_argument("--asr_rolling_seconds", type=float, default=1.0)
    parser.add_argument("--asr_lookahead_logits", type=int, default=10)
    parser.add_argument("--speech_rms_threshold", type=float, default=0.006)
    parser.add_argument("--speech_rms_full", type=float, default=0.05)
    parser.add_argument(
        "--deepspeech_model",
        type=str,
        default="~/.tensorflow/models/deepspeech-0_1_0-b90017e8.pb",
    )
    args = get_combined_args(parser)
    if not 0 <= args.https_port <= 65535:
        raise ValueError("HTTPS port must be 0 (disabled) or a valid TCP port")
    if args.https_port and args.https_port == args.port:
        raise ValueError("HTTP and HTTPS ports must differ")
    if not 1 <= args.fps <= 60:
        raise ValueError("Output render FPS must be between 1 and 60")
    asr_gpu = str(args.asr_gpu).strip().lower()
    if asr_gpu not in ("", "cpu", "none", "-1"):
        try:
            same_gpu = args.render_gpu == int(asr_gpu)
        except ValueError:
            raise ValueError("asr_gpu must be a CUDA index or 'cpu'")
        if same_gpu:
            raise ValueError(
                "render_gpu and asr_gpu must differ for the legacy TensorFlow graph"
            )

    # Start TensorFlow first; its child process selects the ASR GPU before import.
    asr = StreamingDeepSpeechProcess(
        model_path=args.deepspeech_model,
        gpu=args.asr_gpu,
        sample_rate=16000,
        block_ms=args.asr_block_ms,
        rolling_seconds=args.asr_rolling_seconds,
        lookahead_logits=args.asr_lookahead_logits,
    )
    asr.start()

    try:
        dataset = model.extract(args)
        render_pipeline = pipeline.extract(args)
        profiles = {
            "may": load_profile("May", dataset, render_pipeline, args)
        }
        macron_dataset = copy.copy(dataset)
        macron_dataset.source_path = os.path.abspath(args.macron_source)
        macron_dataset.model_path = args.macron_model
        profiles["macron"] = load_profile(
            "Macron", macron_dataset, render_pipeline, args
        )

        service = RealtimeService(
            profiles,
            asr,
            fps=args.fps,
            feature_fps=25.0,
            speech_rms_threshold=args.speech_rms_threshold,
            speech_rms_full=args.speech_rms_full,
        )
        service.start()
        html_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "realtime", "static", "index.html"
        )
        ssl_context = None
        if args.https_port:
            certificate = os.path.abspath(os.path.expanduser(args.ssl_cert))
            private_key = os.path.abspath(os.path.expanduser(args.ssl_key))
            if not os.path.isfile(certificate) or not os.path.isfile(private_key):
                raise FileNotFoundError(
                    "HTTPS certificate or private key is missing"
                )
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(certificate, private_key)
        ca_certificate = (
            os.path.abspath(os.path.expanduser(args.ssl_ca))
            if args.ssl_ca else None
        )
        app = create_app(
            service, html_path, ca_certificate_path=ca_certificate
        )
        asyncio.run(
            run_sites(
                app,
                args.host,
                args.port,
                https_port=args.https_port,
                ssl_context=ssl_context,
            )
        )
    except Exception:
        asr.stop()
        raise


if __name__ == "__main__":
    main()
