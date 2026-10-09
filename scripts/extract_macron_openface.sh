#!/usr/bin/env bash
set -eo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

video="data/macron/macron.mp4"
raw_csv="data/macron/openface/macron.csv"
final_csv="data/macron/au.csv"
expected_frames=8732

mkdir -p data/macron/openface
if [[ ! -s "$final_csv" ]]; then
    FeatureExtraction \
        -f "$video" -out_dir data/macron/openface \
        -aus -q -no2Dfp -no3Dfp -noMparams -noPose -noGaze
fi

python - "$raw_csv" "$expected_frames" <<'PY'
import sys
import numpy as np
import pandas as pd

path, expected = sys.argv[1], int(sys.argv[2])
au = pd.read_csv(path)
au.columns = au.columns.str.strip()
required = ['AU01_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r',
            'AU25_r', 'AU45_r', 'success']
assert len(au) == expected, (len(au), expected)
assert all(column in au for column in required)
assert np.isfinite(au[required].to_numpy()).all()
assert float(au['success'].mean()) >= 0.95
print('[MACRON_OPENFACE] {} frames, tracking success {:.1%}'.format(
    len(au), float(au['success'].mean())))
PY

cp "$raw_csv" "$final_csv"
echo "[MACRON_OPENFACE_COMPLETE] $final_csv"
