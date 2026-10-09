# TalkingGaussian: Structure-Persistent 3D Talking Head Synthesis via Gaussian Splatting

**中文版说明：**[README_CN.md](README_CN.md)（包含本仓库的离线复现、May/Macron 在线演示及实测时延说明）。

This repository builds on the [original TalkingGaussian implementation](https://github.com/Fictionarry/TalkingGaussian) for the ECCV 2024 paper **TalkingGaussian: Structure-Persistent 3D Talking Head Synthesis via Gaussian Splatting**. It adds the training, rendering, and browser-based online experiments described below.

[Paper](https://arxiv.org/abs/2404.15264) | [Project](https://fictionarry.github.io/TalkingGaussian/) | [Video](https://youtu.be/c5VG7HkDs8I)

![image](./assets/main.png)

## Installation

Tested on Ubuntu 18.04, CUDA 11.3, PyTorch 1.12.1

```
git clone git@github.com:Fictionarry/TalkingGaussian.git --recursive

conda env create --file environment.yml
conda activate talking_gaussian
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install tensorflow-gpu==2.8.0
```

If encounter installation problem from the `diff-gaussian-rasterization` or `gridencoder`, please refer to [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [torch-ngp](https://github.com/ashawkey/torch-ngp).

### Preparation

- Prepare face-parsing model and  the 3DMM model for head pose estimation.

  ```bash
  bash scripts/prepare.sh
  ```

- Download 3DMM model from [Basel Face Model 2009](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=details):

  ```bash
  # 1. copy 01_MorphableModel.mat to data_util/face_tracking/3DMM/
  # 2. run following
  cd data_utils/face_tracking
  python convert_BFM.py
  ```

- Prepare the environment for [EasyPortrait](https://github.com/hukenovs/easyportrait):

  ```bash
  # prepare mmcv
  conda activate talking_gaussian
  pip install -U openmim
  mim install mmcv-full==1.7.1

  # download model weight
  cd data_utils/easyportrait
  wget "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/easyportrait/experiments/models/fpn-fp-512.pth"
  ```

## Usage

### Important Notice

- This code is provided for research purposes only. The author makes no warranties, express or implied, as to the accuracy, completeness, or fitness for a particular purpose of the code. Use this code at your own risk.

- The author explicitly prohibits the use of this code for any malicious or illegal activities. By using this code, you agree to comply with all applicable laws and regulations, and you agree not to use it to harm others or to perform any actions that would be considered unethical or illegal.

- The author will not be responsible for any damages, losses, or issues that arise from the use of this code. 

- Users are encouraged to use this code responsibly and ethically.

### Video Dataset
[Here](https://drive.google.com/drive/folders/1E_8W805lioIznqbkvTQHWWi5IFXUG7Er?usp=drive_link) we provide two video clips used in our experiments, which are captured from YouTube. Please respect the original content creators' rights and comply with YouTube’s copyright policies in the usage.

Other used videos can be found from [GeneFace](https://github.com/yerfor/GeneFace) and [AD-NeRF](https://github.com/YudongGuo/AD-NeRF). 


### Pre-processing Training Video

* Put training video under `data/<ID>/<ID>.mp4`.

  The video **must be 25FPS, with all frames containing the talking person**. 
  The resolution should be about 512x512, and duration about 1-5 min.

* Run script to process the video.

  ```bash
  python data_utils/process.py data/<ID>/<ID>.mp4
  ```

* Obtain Action Units
  
  Run `FeatureExtraction` in [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace), rename and move the output CSV file to `data/<ID>/au.csv`.

* Generate tooth masks

  ```bash
  export PYTHONPATH=./data_utils/easyportrait 
  python ./data_utils/easyportrait/create_teeth_mask.py ./data/<ID>
  ```

### Audio Pre-process

In our paper, we use DeepSpeech features for evaluation. 

* DeepSpeech

  ```bash
  python data_utils/deepspeech_features/extract_ds_features.py --input data/<name>.wav # saved to data/<name>.npy
  ```

- HuBERT

  Similar to ER-NeRF, HuBERT is also available. Recommended for situations if the audio is not in English.

  Specify `--audio_extractor hubert` when training and testing.

  ```
  python data_utils/hubert.py --wav data/<name>.wav # save to data/<name>_hu.npy
  ```

### Data Loading

By default, we preload all the data into RAM for acceleration, but the consumption maybe unaffordable (about N x 32GB RAM for about N x 5k frames). 

(Experimental) You can try to set `preload=False` in the `scene/dataset_readers.py - readCamerasFromTransforms(·)` to load the data temporally per iteration to save the cost. As the trade-off, the speed could be slower.

### Train

```bash
# If resources are sufficient, partially parallel is available to speed up the training. See the script.
bash scripts/train_xx.sh data/<ID> output/<project_name> <GPU_ID>
```

For the bundled Macron video, `bash scripts/run_macron_pipeline.sh` validates
the 25 FPS source, reuses the already-extracted matching DeepSpeech features,
preprocesses the video, extracts OpenFace AUs and teeth masks, then trains
mouth, face, and fusion sequentially into `output/macron`. The script uses GPU
1 by default (`MACRON_GPU_ID=0` selects GPU 0), saves resumable checkpoints
every 2,000 mouth/face iterations and every 1,000 fusion iterations, and skips
completed stages when rerun. Run it in a `tmux` session for unattended work;
the current run writes `logs/macron_pipeline.log`.

### Test

```bash
# saved to output/<project_name>/test/ours_None/renders
python synthesize_fuse.py -S data/<ID> -M output/<project_name> --eval  
```

### Novel-view Rendering

Render a fixed virtual-camera offset (angles are in degrees):

```bash
python synthesize_novel_view.py -s data/<ID> -m output/<project_name> \
  --view_mode fixed --yaw 10 --pitch 0 \
  --output output/<project_name>/novel_view/fixed_yaw_10.mp4
```

Switch among several viewpoints and optionally mux the corresponding source
audio into a second `*_with_audio.mp4` file:

```bash
python synthesize_novel_view.py -s data/<ID> -m output/<project_name> \
  --view_mode switch --yaw_sequence=-15,-10,-5,0,5,10,15 \
  --hold_frames 25 --draw_label --with_audio \
  --output output/<project_name>/novel_view/view_switch.mp4
```

Render a smooth left-right camera sweep:

```bash
python synthesize_novel_view.py -s data/<ID> -m output/<project_name> \
  --view_mode sweep --yaw_amplitude 15 --with_audio \
  --output output/<project_name>/novel_view/view_sweep.mp4
```

Monocular training videos only supervise a limited range of viewpoints. Keep
novel yaw/pitch offsets close to the poses observed in the training video.

### Realtime Browser Inference (May and Macron Checkpoints)

The realtime path keeps both `output/may/chkpnt_fuse_latest.pth` and
`output/macron/chkpnt_fuse_latest.pth` unchanged. Both renderers are loaded at
startup; the page switches the active model without restarting DeepSpeech.
A browser sends 16 kHz mono microphone PCM over a
WebSocket; a separate TensorFlow process extracts rolling DeepSpeech features,
and the persistent PyTorch renderer targets 25 FPS avatar frames by default,
matching the checkpoints' DeepSpeech feature time base. Offline videos
are also written at 25 FPS; a higher offline processing FPS is render throughput,
not the encoded video's playback rate. The page reports server render FPS,
browser-displayed FPS, and estimated render capacity separately. The page
receives JPEG frames over a bounded, latest-frame WebSocket video stream to
avoid per-frame HTTP round trips and unbounded video backlog. If the video
WebSocket disconnects, it temporarily polls `/frame.jpg` instead. The page
also provides live yaw, pitch, and zoom controls; the legacy `/video.mjpg`
stream remains available for external tools.
Increasing `--fps` only refreshes the display more often; it does not add new
audio-driven mouth-motion information beyond the 25 FPS feature time base.

This setup uses GPU 0 for Gaussian rendering and GPU 1 for the legacy
DeepSpeech graph. Start it in a `tmux` session from the activated environment:

```bash
conda activate talking_gaussian
tmux new -s tg_dual_realtime
bash scripts/run_realtime_dual.sh
```

The DeepSpeech graph defaults to
`~/.tensorflow/models/deepspeech-0_1_0-b90017e8.pb`. Use
`--deepspeech_model /path/to/model.pb` if it is stored elsewhere. Wait until
`/api/status` reports both `state: ready` and `asr.type: asr_ready` before
speaking.

On this server the launcher binds to the campus interface
`192.168.167.5:8080` by default.  Users connected to the campus network can
open:

```text
http://192.168.167.5:8080/
```

This is one shared research session: avatar changes, camera controls, and audio
input affect every connected viewer.  The campus HTTP origin supports viewing
and the built-in May/Macron validation-audio path, but browsers do not grant
physical microphone access to non-HTTPS remote origins.  For microphone use,
download the dedicated campus-demo CA certificate from
`http://192.168.167.5:8080/campus-ca.crt`, install it for the current user as a
trusted certificate authority, restart the browser, and open:

```text
https://192.168.167.5:8443/
```

The HTTPS leaf certificate is valid for the IP address `192.168.167.5`; its
dedicated CA private key was deleted after signing.  The page displays the CA
SHA-256 fingerprint and per-platform installation guidance.  Remove this CA
from the client's trust store after the demo if it is no longer needed.

For physical microphone access, create an SSH tunnel on the computer running
the browser:

```bash
ssh -L 8080:192.168.167.5:8080 <user>@192.168.167.5
```

The page shows whether the video link is using WebSocket or HTTP fallback.
If a forwarding tool does not support WebSocket upgrades, use the SSH tunnel
above and reload the page.

Then open `http://127.0.0.1:8080` and click **Start microphone**. Localhost is a
browser secure context, so microphone access works without exposing the
research server beyond the campus interface.  To run a private localhost-only
instance instead, set `TG_HOST=127.0.0.1` before launching the script; `TG_PORT`
and `TG_HTTPS_PORT` override ports 8080 and 8443. The first motion needs about one second
of audio; steady-state DeepSpeech context is approximately 480 ms, plus
network, rendering, and display delay. Realtime mode fixes the upper-face AU
instead of cycling unrelated validation expressions, so the avatar remains
still while there is no audio drive. The page can play the local microphone
through a configurable delay for lip-sync inspection. Automatic mode uses the
previous run's measured application-layer median (initial setting: 570 ms),
and locks that setting throughout playback to avoid pitch-shift artifacts.
This setting is not a measured audiovisual synchronization error. Use headphones
to prevent feedback; the delay slider can also be adjusted manually.

The latency panel now traces the actual source audio packet through rolling
DeepSpeech, rendering, video delivery and a browser rAF Canvas submission. Its
total uses two timestamps from the same browser monotonic clock, not a sum of
theoretical context, half-frame waits or RTT/2 estimates. It includes per-frame
measured stages, startup time, median/P95/max and a raw JSON export. Only the
audio-originating page computes this total; other viewers show no invented
cross-clock delay. It excludes pre-callback audio buffering and physical screen
presentation. See [latency measurement definitions and tests](realtime/LATENCY_MEASUREMENT.md).

Select **May** or **Macron** at the top of the page. For an in-distribution
check, **Validation audio · simulated microphone** plays the selected
validation WAV into the same Web Audio processor used by `getUserMedia`:
browser audio-clock samples -> 16 kHz conversion -> PCM WebSocket -> rolling
DeepSpeech -> renderer. There is no separate timer-paced PCM sender or direct
feature injection. Only the source node differs from the physical microphone;
this does not create an operating-system virtual microphone device. May uses
220.8--242.92 s (553 validation frames); Macron uses 317.52--349.28 s
(794 frames). Switching avatars stops the previous input and immediately
re-enables both online input buttons without reloading the page.

The page also embeds both offline validation renders with their aligned source
audio: `output/may/test/ours_None/renders/out_with_audio.mp4` and
`output/macron/test/ours_None/renders/out_with_audio.mp4`. To create a missing
version, run `python scripts/mux_validation_audio.py may` or `... macron`.
These are validation segments, not the full source videos. The browser video
players support seeking via HTTP range requests.

The old DeepSpeech graph has bidirectional context and global MFCC
normalization. One-second rolling inference cannot reproduce the features from
the full 243-second May audio exactly; on sampled May timestamps, the mean
absolute feature difference was about 1.5--5.0. The old saved-feature
reference button was useful only for diagnosing this extractor mismatch, so it
is no longer shown in the browser. The backend diagnostic command remains
available for experiments; it is not evidence of live lip-sync accuracy.

The dashboard reports audio/video payload totals and rates, client counts,
render FPS, audio RMS, speech activity, natural-motion strength, DeepSpeech
feature variation, queue depth, dropped features, camera controls, and the age
of the latest displayed frame and audio-feature queue. A growing age points to
delivery lag. Natural upper-face/pose motion is gated by speech energy: it is
completely static when
idle and smoothly replays low-amplitude motion from the selected model's validation sequence
while speaking. This motion is for visual naturalness and is not phoneme
conditioned. Feature variation is a driver diagnostic, not a lip-sync accuracy
metric. The preserved checkpoint uses legacy English DeepSpeech features, so
Chinese phoneme-to-viseme generalization is not guaranteed.

An end-to-end file-paced test is available without a physical server
microphone:

```bash
python scripts/test_realtime_file.py \
  --url http://127.0.0.1:8080 \
  --wav data/may/aud.wav --start 220.8 --seconds 8 \
  --output_frame output/may/realtime/file_test_frame.jpg
```

For Macron, select it through `POST /api/avatar` with
`{"avatar":"macron"}` and use `data/macron/aud.wav --start 317.52` in the test
command. Both models were trained from monocular videos, so keep live view
offsets modest (the UI limits them by default).

### Inference with Specified Audio

```bash
python synthesize_fuse.py -S data/<ID> -M output/<project_name> --use_train --audio <preprocessed_audio_feature>.npy
```

## Follow-Up 
- [2025/02/28] Our work [InsTaG](https://fictionarry.github.io/InsTaG/) at CVPR 2025 is released! 🔥

## Citation

Consider citing as below if you find this repository helpful to your project:

```
@inproceedings{li2024talkinggaussian,
  title={TalkingGaussian: Structure-Persistent 3D Talking Head Synthesis via Gaussian Splatting},
  author={Li, Jiahe and Zhang, Jiawei and Bai, Xiao and Zheng, Jin and Ning, Xin and Zhou, Jun and Gu, Lin},
  booktitle={European Conference on Computer Vision},
  pages={127--145},
  year={2024},
  organization={Springer}
}
```


## Acknowledgement

This code is developed on [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) with [simple-knn](https://gitlab.inria.fr/bkerbl/simple-knn), and a modified [diff-gaussian-rasterization](https://github.com/ashawkey/diff-gaussian-rasterization). Partial codes are from [RAD-NeRF](https://github.com/ashawkey/RAD-NeRF), [DFRF](https://github.com/sstzal/DFRF), [GeneFace](https://github.com/yerfor/GeneFace), and [AD-NeRF](https://github.com/YudongGuo/AD-NeRF). Teeth mask is from [EasyPortrait](https://github.com/hukenovs/easyportrait). Thanks for these great projects!
