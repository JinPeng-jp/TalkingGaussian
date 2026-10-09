#!/usr/bin/env python
"""Feed a WAV file to realtime_server.py at wall-clock speed."""

import argparse
import asyncio
import json
import os
import time
import wave

from aiohttp import ClientSession


async def capture_jpeg(session, base_url, output_path):
    async with session.get(base_url + "/video.mjpg") as response:
        response.raise_for_status()
        buffer = bytearray()
        while True:
            chunk = await response.content.readany()
            if not chunk:
                raise RuntimeError("MJPEG stream closed before a frame arrived")
            buffer.extend(chunk)
            start = buffer.find(b"\xff\xd8")
            end = buffer.find(b"\xff\xd9", start + 2) if start >= 0 else -1
            if start >= 0 and end >= 0:
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                with open(output_path, "wb") as output:
                    output.write(buffer[start : end + 2])
                return end + 2 - start
            if len(buffer) > 4 * 1024 * 1024:
                del buffer[:-1024]


async def run(args):
    with wave.open(args.wav, "rb") as source:
        if source.getframerate() != 16000 or source.getnchannels() != 1:
            raise ValueError("Test WAV must be 16 kHz mono")
        if source.getsampwidth() != 2:
            raise ValueError("Test WAV must contain signed 16-bit PCM")
        source.setpos(min(source.getnframes(), int(round(args.start * 16000))))
        remaining = int(round(args.seconds * 16000))
        block_frames = int(round(args.block_ms * 16000 / 1000.0))

        websocket_url = args.url.replace("http://", "ws://").replace(
            "https://", "wss://"
        ) + "/ws/audio"
        async with ClientSession() as session:
            async with session.ws_connect(websocket_url, max_msg_size=2 ** 20) as socket:
                await socket.send_str(
                    json.dumps({"command": "reset", "source": args.source})
                )
                started = time.perf_counter()
                sent = 0
                while remaining > 0:
                    count = min(block_frames, remaining)
                    payload = source.readframes(count)
                    if not payload:
                        break
                    await socket.send_bytes(payload)
                    sent += len(payload) // 2
                    remaining -= len(payload) // 2
                    target = started + sent / 16000.0
                    await asyncio.sleep(max(0.0, target - time.perf_counter()))

            await asyncio.sleep(args.settle)
            async with session.get(args.url + "/api/status") as response:
                status = await response.json()
            jpeg_size = await capture_jpeg(session, args.url, args.output_frame)
            print(json.dumps(status, indent=2, ensure_ascii=False))
            print("captured_jpeg={} bytes path={}".format(jpeg_size, args.output_frame))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--wav", default="data/may/aud.wav")
    parser.add_argument("--start", type=float, default=220.8)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--block_ms", type=int, default=80)
    parser.add_argument("--settle", type=float, default=0.5)
    parser.add_argument(
        "--source",
        choices=("dataset_audio", "reference_features", "may", "may_reference", "microphone", "websocket"),
        default="dataset_audio",
        help="Source label reported by the realtime status endpoint",
    )
    parser.add_argument(
        "--output_frame", default="output/may/realtime/file_test_frame.jpg"
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
