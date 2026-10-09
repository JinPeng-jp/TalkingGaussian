#!/usr/bin/env python
"""Check that a preprocessed TalkingGaussian Macron dataset is trainable."""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("expected_frames", type=int)
    args = parser.parse_args()
    dataset = args.dataset
    expected = args.expected_frames

    features = np.load(os.path.join(dataset, "aud_ds.npy"), mmap_mode="r")
    if features.shape != (expected, 16, 29) or not np.isfinite(features).all():
        raise ValueError("DeepSpeech features are missing, misaligned, or non-finite")

    au = pd.read_csv(os.path.join(dataset, "au.csv"))
    au.columns = au.columns.str.strip()
    required_au = ["AU01_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
                   "AU25_r", "AU45_r", "success"]
    if len(au) != expected or any(column not in au for column in required_au):
        raise ValueError("OpenFace AU rows or columns do not match the video")
    success = float(au["success"].mean())
    if success < 0.95 or not np.isfinite(au[required_au].to_numpy()).all():
        raise ValueError("OpenFace tracking success is below 95% or contains non-finite AUs")

    tracker = torch.load(os.path.join(dataset, "track_params.pt"), map_location="cpu")
    for key, width in (("euler", 3), ("trans", 3)):
        value = tracker[key]
        if tuple(value.shape) != (expected, width) or not torch.isfinite(value).all():
            raise ValueError("3D face tracking field {} is invalid".format(key))

    frame_ids = []
    for split in ("train", "val"):
        with open(os.path.join(dataset, "transforms_{}.json".format(split))) as source:
            frames = json.load(source)["frames"]
        if not frames:
            raise ValueError("{} split is empty".format(split))
        frame_ids.extend(int(frame["img_id"]) for frame in frames)
        for frame in frames:
            if int(frame["aud_id"]) != int(frame["img_id"]):
                raise ValueError("Audio/video frame IDs differ")
            if not np.isfinite(np.asarray(frame["transform_matrix"])).all():
                raise ValueError("A camera transform contains non-finite values")
        print("[MACRON] {} split: {} frames".format(split, len(frames)))
    if frame_ids != list(range(expected)):
        raise ValueError("Camera frame IDs are not contiguous from 0")

    for index in range(expected):
        for folder, suffix in (("ori_imgs", ".jpg"), ("ori_imgs", ".lms"),
                               ("parsing", ".png"), ("gt_imgs", ".jpg"),
                               ("torso_imgs", ".png"), ("teeth_mask", ".npy")):
            path = os.path.join(dataset, folder, "{}{}".format(index, suffix))
            if not os.path.isfile(path) or os.path.getsize(path) == 0:
                raise ValueError("Missing or empty preprocessing file: {}".format(path))

    print("[MACRON] Dataset validated: {} frames, OpenFace success {:.1%}".format(
        expected, success))


if __name__ == "__main__":
    main()
