#!/usr/bin/env python

import glob
import os
import re
import sys

import torch


workspace, pattern = sys.argv[1:3]
candidates = []
for path in glob.glob(os.path.join(workspace, pattern)):
    match = re.search(r"_(\d+)\.pth$", path)
    if match:
        candidates.append((int(match.group(1)), path))

for expected_iteration, path in sorted(candidates, reverse=True):
    try:
        checkpoint = torch.load(path, map_location="cpu")
        if checkpoint[-1] == expected_iteration:
            print(path)
            break
    except Exception as error:
        print("[WARN] Ignoring invalid checkpoint {}: {}".format(path, error), file=sys.stderr)
