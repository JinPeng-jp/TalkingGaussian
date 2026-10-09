"""Streaming DeepSpeech features compatible with the released May checkpoint.

The training pipeline uses 29-dimensional DeepSpeech logits, 16-logit windows,
and an eight-window attention context.  This module preserves that exact tensor
shape while evaluating the legacy frozen graph on a rolling audio buffer.

TensorFlow is deliberately imported only inside the worker process.  The old
graph reserves substantial GPU memory, so the online server can place it on a
different GPU from the PyTorch Gaussian renderer.
"""

import multiprocessing as mp
import os
import queue
import time

import numpy as np

from realtime.latency_trace import AudioTraceTimeline


def _put_latest(target_queue, value):
    """Put without blocking, dropping the oldest queued item when necessary."""
    try:
        target_queue.put_nowait(value)
        return
    except queue.Full:
        pass
    try:
        target_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        target_queue.put_nowait(value)
    except queue.Full:
        pass


def _make_feature_payload(features, rms, previous_features):
    """Attach observable driver statistics to a feature tensor."""
    features = features.astype(np.float32, copy=False)
    if previous_features is None:
        delta = 0.0
    else:
        delta = float(np.mean(np.abs(features - previous_features)))
    payload = {
        "features": features,
        "rms": float(rms),
        "feature_delta": delta,
        "feature_std": float(np.std(features)),
        "feature_mean": float(np.mean(features)),
    }
    return payload, features.copy()


def _worker_main(
    audio_queue,
    feature_queue,
    status_queue,
    model_path,
    gpu,
    sample_rate,
    block_samples,
    rolling_seconds,
    lookahead_logits,
):
    if gpu is None or str(gpu).lower() in ("", "cpu", "none", "-1"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        use_gpu = False
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        use_gpu = True
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

    # These imports must remain after CUDA_VISIBLE_DEVICES is assigned.
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    from data_utils.deepspeech_features.deepspeech_features import (
        conv_audio_to_deepspeech_input_vector,
        prepare_deepspeech_net,
    )

    graph = tf.Graph()
    with graph.as_default():
        _, logits_tensor, input_tensor, lengths_tensor = prepare_deepspeech_net(
            model_path
        )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if not use_gpu:
        config.device_count["GPU"] = 0

    rolling_samples = int(round(sample_rate * rolling_seconds))
    if rolling_samples < sample_rate:
        raise ValueError("rolling_seconds must be at least 1.0")
    if block_samples <= 0:
        raise ValueError("block_samples must be positive")

    pcm_pending = np.empty((0,), dtype=np.int16)
    rolling_pcm = np.empty((0,), dtype=np.int16)
    attention_windows = []
    attention_centers = []
    timeline = AudioTraceTimeline(rolling_samples * 4)
    initialized = False
    audio_samples_seen = 0
    emitted = 0
    inference_times = []
    previous_features = None
    latest_feature_delta = 0.0
    latest_feature_std = 0.0
    latest_feature_mean = 0.0

    with tf.Session(graph=graph, config=config) as session:
        # Force graph placement, cuDNN/cuBLAS initialization and allocator setup
        # before a browser is allowed to send audio.  Without this warm-up the
        # first live utterance can fill the input queue while TensorFlow starts.
        warm_pcm = np.zeros((rolling_samples,), dtype=np.int16)
        warm_input = conv_audio_to_deepspeech_input_vector(
            warm_pcm,
            sample_rate=sample_rate,
            num_cepstrum=26,
            num_context=9,
        )
        warm_input = np.nan_to_num(
            warm_input, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32, copy=False)
        warm_times = []
        for _ in range(3):
            warm_started = time.perf_counter()
            session.run(
                logits_tensor,
                feed_dict={
                    input_tensor: warm_input[np.newaxis, ...],
                    lengths_tensor: [warm_input.shape[0]],
                },
            )
            warm_times.append((time.perf_counter() - warm_started) * 1000.0)
        _put_latest(
            status_queue,
            {
                "type": "asr_ready",
                "gpu": str(gpu),
                "rolling_seconds": rolling_seconds,
                "block_ms": 1000.0 * block_samples / sample_rate,
                "warmup_ms": [round(value, 2) for value in warm_times],
            },
        )

        while True:
            message = audio_queue.get()
            if isinstance(message, tuple):
                command, payload = message
            else:
                command, payload = "audio", message

            if command == "stop":
                break
            if command == "reset":
                pcm_pending = np.empty((0,), dtype=np.int16)
                rolling_pcm = np.empty((0,), dtype=np.int16)
                attention_windows = []
                attention_centers = []
                timeline = AudioTraceTimeline(rolling_samples * 4)
                initialized = False
                audio_samples_seen = 0
                emitted = 0
                previous_features = None
                latest_feature_delta = 0.0
                latest_feature_std = 0.0
                latest_feature_mean = 0.0
                _put_latest(status_queue, {"type": "asr_reset"})
                continue
            if command != "audio" or not payload:
                continue

            trace_metadata = None
            if isinstance(payload, dict):
                trace_metadata = payload.get("trace")
                payload = payload["pcm"]
            incoming = np.frombuffer(payload, dtype="<i2").astype(np.int16, copy=True)
            if incoming.size == 0:
                continue
            timeline.append(incoming.size, trace_metadata)
            pcm_pending = np.concatenate((pcm_pending, incoming))

            while pcm_pending.size >= block_samples:
                block = pcm_pending[:block_samples]
                pcm_pending = pcm_pending[block_samples:]
                audio_samples_seen += block.size
                rolling_pcm = np.concatenate((rolling_pcm, block))
                if rolling_pcm.size > rolling_samples:
                    rolling_pcm = rolling_pcm[-rolling_samples:]
                if rolling_pcm.size < rolling_samples:
                    continue

                rms = float(
                    np.sqrt(np.mean(block.astype(np.float32) ** 2) + 1e-12)
                    / 32768.0
                )

                started = time.perf_counter()
                asr_started = time.monotonic()
                network_input = conv_audio_to_deepspeech_input_vector(
                    rolling_pcm,
                    sample_rate=sample_rate,
                    num_cepstrum=26,
                    num_context=9,
                )
                network_input = np.nan_to_num(
                    network_input, nan=0.0, posinf=0.0, neginf=0.0
                ).astype(np.float32, copy=False)
                logits = session.run(
                    logits_tensor,
                    feed_dict={
                        input_tensor: network_input[np.newaxis, ...],
                        lengths_tensor: [network_input.shape[0]],
                    },
                )[:, 0, :]
                asr_finished = time.monotonic()
                elapsed = time.perf_counter() - started
                inference_times.append(elapsed)
                inference_times = inference_times[-50:]

                latest_start = logits.shape[0] - lookahead_logits - 16
                if latest_start < 0:
                    continue

                windows_per_block = max(
                    1, int(round(block_samples / float(sample_rate / 25.0)))
                )
                if not initialized:
                    starts = list(range(0, latest_start + 1, 2))
                    seed = [logits[start : start + 16] for start in starts[-8:]]
                    if len(seed) < 8:
                        continue
                    attention_windows = seed
                    attention_centers = [
                        audio_samples_seen - rolling_samples +
                        (start + 8) * sample_rate / 50.0
                        for start in starts[-8:]
                    ]
                    features = np.stack(attention_windows, axis=0).transpose(0, 2, 1)
                    payload, previous_features = _make_feature_payload(
                        features, rms, previous_features
                    )
                    payload["audio_seconds"] = audio_samples_seen / float(sample_rate)
                    payload["generated_at"] = time.monotonic()
                    payload["trace"] = timeline.feature_trace(
                        attention_centers[4], emitted, asr_started, asr_finished
                    )
                    latest_feature_delta = payload["feature_delta"]
                    latest_feature_std = payload["feature_std"]
                    latest_feature_mean = payload["feature_mean"]
                    _put_latest(feature_queue, payload)
                    emitted += 1
                    initialized = True
                else:
                    starts = [
                        latest_start - 2 * offset
                        for offset in reversed(range(windows_per_block))
                    ]
                    for start in starts:
                        if start < 0:
                            continue
                        attention_windows.append(logits[start : start + 16])
                        attention_windows = attention_windows[-8:]
                        attention_centers.append(
                            audio_samples_seen - rolling_samples +
                            (start + 8) * sample_rate / 50.0
                        )
                        attention_centers = attention_centers[-8:]
                        if len(attention_windows) == 8:
                            features = np.stack(attention_windows, axis=0).transpose(
                                0, 2, 1
                            )
                            payload, previous_features = _make_feature_payload(
                                features, rms, previous_features
                            )
                            payload["audio_seconds"] = audio_samples_seen / float(sample_rate)
                            payload["generated_at"] = time.monotonic()
                            payload["trace"] = timeline.feature_trace(
                                attention_centers[4], emitted, asr_started, asr_finished
                            )
                            latest_feature_delta = payload["feature_delta"]
                            latest_feature_std = payload["feature_std"]
                            latest_feature_mean = payload["feature_mean"]
                            _put_latest(feature_queue, payload)
                            emitted += 1

                _put_latest(
                    status_queue,
                    {
                        "type": "asr_progress",
                        "audio_seconds": audio_samples_seen / float(sample_rate),
                        "feature_frames": emitted,
                        "inference_ms": elapsed * 1000.0,
                        "mean_inference_ms": float(np.mean(inference_times)) * 1000.0,
                        "rms": rms,
                        "feature_delta": latest_feature_delta,
                        "feature_std": latest_feature_std,
                        "feature_mean": latest_feature_mean,
                    },
                )

    _put_latest(status_queue, {"type": "asr_stopped"})


class StreamingDeepSpeechProcess:
    """Parent-side controller for the isolated TensorFlow worker."""

    def __init__(
        self,
        model_path,
        gpu="1",
        sample_rate=16000,
        block_ms=80,
        rolling_seconds=1.0,
        lookahead_logits=10,
    ):
        self.model_path = os.path.abspath(os.path.expanduser(model_path))
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(self.model_path)
        self.gpu = gpu
        self.sample_rate = int(sample_rate)
        self.block_samples = int(round(self.sample_rate * block_ms / 1000.0))
        self.rolling_seconds = float(rolling_seconds)
        self.lookahead_logits = int(lookahead_logits)

        context = mp.get_context("spawn")
        self.audio_queue = context.Queue(maxsize=256)
        self.feature_queue = context.Queue(maxsize=64)
        self.status_queue = context.Queue(maxsize=16)
        self.process = context.Process(
            target=_worker_main,
            args=(
                self.audio_queue,
                self.feature_queue,
                self.status_queue,
                self.model_path,
                self.gpu,
                self.sample_rate,
                self.block_samples,
                self.rolling_seconds,
                self.lookahead_logits,
            ),
            name="talking-gaussian-deepspeech",
        )
        self.process.daemon = True

    def start(self):
        self.process.start()

    def push_pcm(self, pcm_bytes, trace=None):
        if not pcm_bytes:
            return
        _put_latest(self.audio_queue, ("audio", {
            "pcm": bytes(pcm_bytes), "trace": trace,
        }))

    def reset(self):
        _put_latest(self.audio_queue, ("reset", None))
        while True:
            try:
                self.feature_queue.get_nowait()
            except queue.Empty:
                break

    def stop(self, timeout=10.0):
        if not self.process.is_alive():
            return
        _put_latest(self.audio_queue, ("stop", None))
        self.process.join(timeout)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(2.0)

    def is_alive(self):
        return self.process.is_alive()
