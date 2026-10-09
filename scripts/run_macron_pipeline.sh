#!/usr/bin/env bash
set -Eeo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if [[ "${CONDA_DEFAULT_ENV:-}" != "talking_gaussian" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate talking_gaussian
fi
set -u

dataset="data/macron"
workspace="output/macron"
video="$dataset/macron.mp4"
gpu_id="${MACRON_GPU_ID:-1}"
export CUDA_VISIBLE_DEVICES="$gpu_id"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONUNBUFFERED=1

trap 'rc=$?; echo "[MACRON_ERROR] line $LINENO, exit $rc"; exit "$rc"' ERR

test -s "$video"
video_info="$(env -u LD_LIBRARY_PATH /usr/bin/ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height,avg_frame_rate,nb_frames \
    -of csv=p=0 "$video")"
IFS=, read -r width height video_fps expected_frames <<< "$video_info"
if [[ "$width" != 512 || "$height" != 512 || "$video_fps" != 25/1 || \
      ! "$expected_frames" =~ ^[0-9]+$ ]]; then
    echo "[MACRON_ERROR] Expected a 512x512, 25 FPS video with known frame count: $video_info"
    exit 1
fi
echo "[MACRON] source=$video frames=$expected_frames GPU=$gpu_id"

count_files() {
    local directory="$1" pattern="$2"
    if [[ ! -d "$directory" ]]; then
        echo 0
    else
        find "$directory" -maxdepth 1 -type f -name "$pattern" | wc -l
    fi
}

prepare_counted_stage() {
    local label="$1" task="$2" directory="$3" pattern="$4" count
    count="$(count_files "$directory" "$pattern")"
    if (( count != expected_frames )); then
        echo "[MACRON] $label: $count/$expected_frames, running process task $task"
        python data_utils/process.py "$video" --task "$task"
        count="$(count_files "$directory" "$pattern")"
    fi
    if (( count != expected_frames )); then
        echo "[MACRON_ERROR] $label incomplete: $count/$expected_frames"
        exit 1
    fi
    echo "[MACRON] $label complete: $count/$expected_frames"
}

# These were previously extracted from the same hard-linked author video.
if [[ ! -s "$dataset/aud.wav" ]]; then
    cp data/custom_audio/macron/macron_16k_mono.wav "$dataset/aud.wav"
fi
if [[ ! -s "$dataset/aud_ds.npy" ]]; then
    cp data/custom_audio/macron/macron_ds_25fps.npy "$dataset/aud_ds.npy"
fi
python - "$dataset/aud.wav" "$dataset/aud_ds.npy" "$expected_frames" <<'PY'
import sys
import wave
import numpy as np

wav_path, feature_path, frames = sys.argv[1], sys.argv[2], int(sys.argv[3])
features = np.load(feature_path, mmap_mode='r')
assert features.shape == (frames, 16, 29), features.shape
with wave.open(wav_path, 'rb') as audio:
    assert audio.getframerate() == 16000 and audio.getnchannels() == 1
    assert audio.getsampwidth() == 2
    assert abs(audio.getnframes() / 16000.0 - frames / 25.0) < 0.1
print('[MACRON] Audio and DeepSpeech features align with {} video frames'.format(frames))
PY

prepare_counted_stage images 3 "$dataset/ori_imgs" '*.jpg'
prepare_counted_stage parsing 4 "$dataset/parsing" '*.png'

if [[ ! -s "$dataset/bc.jpg" ]]; then
    echo "[MACRON] Estimating static background"
    python data_utils/process.py "$video" --task 5
fi
test -s "$dataset/bc.jpg"

gt_count="$(count_files "$dataset/gt_imgs" '*.jpg')"
torso_count="$(count_files "$dataset/torso_imgs" '*.png')"
if (( gt_count != expected_frames || torso_count != expected_frames )); then
    echo "[MACRON] Compositing torso and ground-truth frames"
    python data_utils/process.py "$video" --task 6
fi
prepare_counted_stage ground_truth 6 "$dataset/gt_imgs" '*.jpg'
prepare_counted_stage torso 6 "$dataset/torso_imgs" '*.png'

prepare_counted_stage landmarks 7 "$dataset/ori_imgs" '*.lms'
if [[ ! -s "$dataset/track_params.pt" ]]; then
    echo "[MACRON] Tracking 3D face pose"
    python data_utils/process.py "$video" --task 8
fi
test -s "$dataset/track_params.pt"

if [[ ! -s "$dataset/transforms_train.json" || ! -s "$dataset/transforms_val.json" ]]; then
    echo "[MACRON] Writing training/validation camera transforms"
    python data_utils/process.py "$video" --task 9
fi

if [[ ! -s "$dataset/au.csv" ]]; then
    mkdir -p "$dataset/openface"
    if [[ ! -s "$dataset/openface/macron.csv" ]]; then
        echo "[MACRON] Extracting OpenFace action units"
        FeatureExtraction -f "$video" -out_dir "$dataset/openface" \
            -aus -q -no2Dfp -no3Dfp -noMparams -noPose -noGaze
    fi
    cp "$dataset/openface/macron.csv" "$dataset/au.csv"
fi

teeth_count="$(count_files "$dataset/teeth_mask" '*.npy')"
if (( teeth_count != expected_frames )); then
    echo "[MACRON] Generating teeth masks: $teeth_count/$expected_frames"
    PYTHONPATH=./data_utils/easyportrait \
        python data_utils/easyportrait/create_teeth_mask.py "$dataset"
fi
teeth_count="$(count_files "$dataset/teeth_mask" '*.npy')"
if (( teeth_count != expected_frames )); then
    echo "[MACRON_ERROR] Teeth masks incomplete: $teeth_count/$expected_frames"
    exit 1
fi
echo "[MACRON] Teeth masks complete: $teeth_count/$expected_frames"

python scripts/validate_macron_dataset.py "$dataset" "$expected_frames"

latest="$(python scripts/find_latest_checkpoint.py "$workspace" 'chkpnt_mouth_[0-9]*.pth')"
if [[ "$latest" != "$workspace/chkpnt_mouth_50000.pth" ]]; then
    echo "[MACRON] Training mouth on GPU $gpu_id"
    bash scripts/train_mouth_resumable.sh "$dataset" "$workspace" "$gpu_id"
fi
test "$(python scripts/find_latest_checkpoint.py "$workspace" 'chkpnt_mouth_[0-9]*.pth')" = \
    "$workspace/chkpnt_mouth_50000.pth"

latest="$(python scripts/find_latest_checkpoint.py "$workspace" 'chkpnt_face_[0-9]*.pth')"
if [[ "$latest" != "$workspace/chkpnt_face_50000.pth" ]]; then
    echo "[MACRON] Training face on GPU $gpu_id"
    bash scripts/train_face_resumable.sh "$dataset" "$workspace" "$gpu_id"
fi
test "$(python scripts/find_latest_checkpoint.py "$workspace" 'chkpnt_face_[0-9]*.pth')" = \
    "$workspace/chkpnt_face_50000.pth"

latest="$(python scripts/find_latest_checkpoint.py "$workspace" 'chkpnt_fuse_resume_[0-9]*.pth')"
if [[ "$latest" != "$workspace/chkpnt_fuse_resume_10000.pth" ]]; then
    echo "[MACRON] Training fusion on GPU $gpu_id"
    bash scripts/train_fuse_resumable.sh "$dataset" "$workspace" "$gpu_id"
fi
test "$(python scripts/find_latest_checkpoint.py "$workspace" 'chkpnt_fuse_resume_[0-9]*.pth')" = \
    "$workspace/chkpnt_fuse_resume_10000.pth"
test -s "$workspace/chkpnt_fuse_latest.pth"

if [[ ! -s "$workspace/test/ours_None/renders/out.mp4" ]]; then
    echo "[MACRON] Rendering validation video"
    python synthesize_fuse.py -s "$dataset" -m "$workspace" --eval \
        --audio_extractor deepspeech
fi

echo "[MACRON_COMPLETE] checkpoint=$workspace/chkpnt_fuse_latest.pth"
