#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
input_video="$repo_dir/data/author_samples/macron.mp4"
audio_dir="$repo_dir/data/custom_audio/macron"
archive_dir="$repo_dir/output/may/custom_audio/macron"
raw_features="$audio_dir/macron_ds_extracted.npy"
features="$audio_dir/macron_ds_25fps.npy"
wav="$audio_dir/macron_16k_mono.wav"
rendered="$repo_dir/output/may/train/ours_None/renders/out.mp4"

mkdir -p "$audio_dir" "$archive_dir"

env -u LD_LIBRARY_PATH /usr/bin/ffmpeg -nostdin -y -v warning \
    -i "$input_video" -vn -acodec pcm_s16le -ac 1 -ar 16000 "$wav"

python data_utils/deepspeech_features/extract_ds_features.py \
    --input "$wav" --output "$raw_features"

# Despite the internal `video_fps = 50` name, the bundled DeepSpeech pipeline
# emits one window per 25 FPS video frame. Keep every extracted window, matching
# the observed convention of data/may/aud_ds.npy.
python - "$raw_features" "$features" <<'PY'
import sys
import numpy as np

source, target = sys.argv[1:]
features = np.load(source)
assert features.ndim == 3 and features.shape[1:] == (16, 29), features.shape
np.save(target, features)
print(f"DeepSpeech features at render rate: {features.shape}")
PY

python synthesize_fuse.py \
    -s data/may -m output/may \
    --use_train --audio "$features" \
    --audio_extractor deepspeech --fast

cp "$rendered" "$archive_dir/may_driven_by_macron_silent.mp4"

env -u LD_LIBRARY_PATH /usr/bin/ffmpeg -nostdin -y -v warning \
    -i "$archive_dir/may_driven_by_macron_silent.mp4" \
    -i "$wav" \
    -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 192k -shortest \
    "$archive_dir/may_driven_by_macron.mp4"

echo "[CUSTOM_AUDIO_OUTPUT] $archive_dir/may_driven_by_macron.mp4"
