#!/usr/bin/env python3
"""Mux a dataset's aligned validation audio into its offline render."""

import argparse
import json
import os
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("avatar", choices=("may", "macron"))
    args = parser.parse_args()

    dataset = os.path.join("data", args.avatar)
    render_dir = os.path.join(
        "output", args.avatar, "test", "ours_None", "renders"
    )
    video = os.path.join(render_dir, "out.mp4")
    audio = os.path.join(dataset, "aud.wav")
    destination = os.path.join(render_dir, "out_with_audio.mp4")
    if not os.path.isfile(video) or not os.path.isfile(audio):
        raise FileNotFoundError("Validation video or audio is missing")
    if os.path.exists(destination):
        print("Already exists: {}".format(destination))
        return

    with open(os.path.join(dataset, "transforms_val.json")) as source:
        frames = json.load(source)["frames"]
    if not frames:
        raise ValueError("Validation split is empty")
    start = int(frames[0]["img_id"]) / 25.0
    duration = len(frames) / 25.0
    if any(int(frame["img_id"]) != int(frames[0]["img_id"]) + index
           for index, frame in enumerate(frames)):
        raise ValueError("Validation frames are not contiguous")

    environment = os.environ.copy()
    environment.pop("LD_LIBRARY_PATH", None)
    subprocess.run([
        "/usr/bin/ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-i", video, "-ss", "{:.6f}".format(start),
        "-t", "{:.6f}".format(duration), "-i", audio,
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
        "-c:a", "aac", "-b:a", "160k", "-shortest",
        "-movflags", "+faststart", destination,
    ], check=True, env=environment)
    print("Muxed {} frames, {:.2f}–{:.2f}s: {}".format(
        len(frames), start, start + duration, destination
    ))


if __name__ == "__main__":
    main()
