#!/usr/bin/env bash

set -eo pipefail

dataset="${1:-data/may}"
workspace="${2:-output/may}"
gpu_id="${3:-0}"

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if [[ "${CONDA_DEFAULT_ENV:-}" != "talking_gaussian" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate talking_gaussian
fi

checkpoint_args=()
latest_checkpoint="$(python scripts/find_latest_checkpoint.py "$workspace" 'chkpnt_face_[0-9]*.pth')"
if [[ -n "$latest_checkpoint" ]]; then
    echo "[INFO] Resuming face training from $latest_checkpoint"
    checkpoint_args=(--start_checkpoint "$latest_checkpoint")
else
    echo "[INFO] No face checkpoint found; starting from iteration 0"
fi

checkpoint_iterations=()
for ((iteration = 2000; iteration <= 50000; iteration += 2000)); do
    checkpoint_iterations+=("$iteration")
done

export CUDA_VISIBLE_DEVICES="$gpu_id"
# TensorFlow 2.8 was built against an older protobuf ABI. TensorBoard only
# needs the Python implementation here, which avoids a native ABI conflict.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python train_face.py \
    -s "$dataset" \
    -m "$workspace" \
    --init_num 2000 \
    --densify_grad_threshold 0.0005 \
    --audio_extractor deepspeech \
    --checkpoint_iterations "${checkpoint_iterations[@]}" \
    "${checkpoint_args[@]}"
