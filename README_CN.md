# TalkingGaussian：离线复现与音频在线驱动验证台

本仓库基于论文 [TalkingGaussian: Structure-Persistent 3D Talking Head Synthesis via Gaussian Splatting](https://arxiv.org/abs/2404.15264) 的[原作者代码](https://github.com/Fictionarry/TalkingGaussian)，保留原有的嘴部、脸部、融合训练和离线渲染流程，并加入 May、Macron 两个化身的浏览器在线驱动、双模型验证、视角控制、可恢复训练及应用层时延测量。英文原版说明见 [README.md](README.md)。论文归属、引用与原作者链接见文末。

> 这里公开的是**源码与脚本**。训练视频、预处理数据、Basel Face Model、DeepSpeech 模型、May/Macron 检查点、离线视频及本机 HTTPS 证书均不包含在 Git 仓库中。仅克隆代码不能直接启动已训练的化身；需要按下文获取数据并训练，或自行放入有使用权限的检查点。

## 1. 获取代码及环境

```bash
git clone --recurse-submodules https://github.com/JinPeng-jp/TalkingGaussian.git
cd TalkingGaussian
conda env create -f environment.yml
conda activate talking_gaussian
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install tensorflow-gpu==2.8.0
```

原项目测试配置为 Ubuntu 18.04、CUDA 11.3、PyTorch 1.12.1；`environment.yml` 使用 Python 3.7，并包括在线服务所需的 `aiohttp`。`simple-knn` 和 `diff-gaussian-rasterization` 是 Git 子模块，所以克隆时需带 `--recurse-submodules`。如果已经普通克隆，可在项目目录执行 `git submodule update --init --recursive`。CUDA 扩展、PyTorch3D 与本机驱动版本需要匹配。

在线版默认把高斯渲染和 DeepSpeech 分别放在 GPU 0、GPU 1。只有一块 GPU 时，可将 `--asr_gpu cpu` 交给 `realtime_server.py`，但需要自行评估 CPU 推理能否跟上音频；启动脚本的默认双 GPU 配置并不适用于单 GPU 机器。

## 2. 准备视频与预处理依赖

原作者提供的实验视频见[数据集链接](https://drive.google.com/drive/folders/1E_8W805lioIznqbkvTQHWWi5IFXUG7Er?usp=drive_link)。请遵守原视频的使用条件。训练视频放在 `data/<ID>/<ID>.mp4`，要求人物始终在画面内、25 FPS，建议约 512×512、1–5 分钟。May 与 Macron 分别使用 `data/may/may.mp4`、`data/macron/macron.mp4`。

运行原项目依赖准备脚本，并从 [Basel Face Model 2009](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=details) 申请模型文件：

```bash
bash scripts/prepare.sh
# 完成许可申请后，自行复制 01_MorphableModel.mat 到：
# data_utils/face_tracking/3DMM/01_MorphableModel.mat
cd data_utils/face_tracking
python convert_BFM.py
cd ../..
```

还需安装 [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) 的 `FeatureExtraction`，以及牙齿掩码依赖 [EasyPortrait](https://github.com/hukenovs/easyportrait)：

```bash
pip install -U openmim
mim install mmcv-full==1.7.1
# 下载 fpn-fp-512.pth 到 data_utils/easyportrait/
```

权重下载链接和详细依赖见[原英文说明](README.md#preparation)。DeepSpeech 0.1.0 冻结图默认为 `~/.tensorflow/models/deepspeech-0_1_0-b90017e8.pb`；预处理和在线服务都需要这个文件。请自行取得并放到该位置。Basel Face Model、第三方权重和数据文件不随本仓库发布。

普通视频的预处理流程：

```bash
python data_utils/process.py data/<ID>/<ID>.mp4
FeatureExtraction -f data/<ID>/<ID>.mp4 -out_dir data/<ID>/openface -aus -q
# 将输出 CSV 复制为 data/<ID>/au.csv，并核对行数与帧数相同。
PYTHONPATH=./data_utils/easyportrait \
  python data_utils/easyportrait/create_teeth_mask.py data/<ID>
```

`process.py` 默认用 DeepSpeech 提取音频特征，并产生图像、人脸分割、跟踪、相机参数等训练文件。运行前请确认 OpenFace、3DMM、脸部解析权重及 DeepSpeech 图都已准备好。Macron 的 `scripts/run_macron_pipeline.sh` 是**针对本实验现有目录结构**的自动脚本：它要求 `data/macron/macron.mp4` 和对应的 `data/custom_audio/macron/macron_16k_mono.wav`、`macron_ds_25fps.npy` 已存在；不会从本仓库下载这些受版权约束的文件。

## 3. 可恢复训练与离线验证

对已经预处理好的数据，依次训练嘴部、脸部和融合模型：

```bash
# 第三个参数为 GPU 编号；例子使用 GPU 0。
bash scripts/train_mouth_resumable.sh data/may output/may 0
bash scripts/train_face_resumable.sh  data/may output/may 0
bash scripts/train_fuse_resumable.sh  data/may output/may 0

python synthesize_fuse.py -s data/may -m output/may --eval \
  --audio_extractor deepspeech
python scripts/mux_validation_audio.py may
```

这三个包装脚本会查找已有检查点并续训；嘴部和脸部每 2,000 次迭代保存检查点，融合阶段每 1,000 次保存一次。服务器长时间训练建议放进 `tmux`。原作者的一键脚本 `bash scripts/train_xx.sh data/<ID> output/<项目名> <GPU_ID>` 也保留，但自动续训请使用上面的脚本。Macron 的实验专用流水线可在准备好相应输入后执行 `bash scripts/run_macron_pipeline.sh`，默认使用 GPU 1，可用 `MACRON_GPU_ID=0` 调整。

离线结果默认写入 `output/<项目名>/test/ours_None/renders/out.mp4`；配音视频由 `scripts/mux_validation_audio.py` 生成 `out_with_audio.mp4`。离线视频有 25 FPS 的播放帧率；推理过程的“每秒处理帧数”是另一项指标。

不同视角渲染的例子：

```bash
python synthesize_novel_view.py -s data/may -m output/may \
  --view_mode fixed --yaw 10 --pitch 0 \
  --output output/may/novel_view/fixed_yaw_10.mp4
```

单目训练视频只覆盖有限视角；偏离训练视角较大时画质可能下降。`synthesize_novel_view.py` 还支持 `switch`、`sweep` 模式，详见[英文说明](README.md#novel-view-rendering)。

## 4. 在线验证台

在线流程为：浏览器麦克风或验证音频 → Web Audio 按块采样 → 16 kHz PCM → 音频 WebSocket → 独立 DeepSpeech 进程 → 高斯渲染 → 视频 WebSocket → 浏览器 Canvas。May 和 Macron 切换的是各自训练好的模型；当前两套检查点都使用旧版英语 DeepSpeech 特征，不保证中文语音的口型准确性。

在线服务需要 `output/may/chkpnt_fuse_latest.pth`、`output/macron/chkpnt_fuse_latest.pth`，两套对应的 `data/<ID>/` 预处理结果，以及上述 DeepSpeech 冻结图。默认渲染使用 GPU 0，DeepSpeech 使用 GPU 1。对只有本机访问需求的环境，可先关闭 HTTPS，在本机启动：

```bash
conda activate talking_gaussian
TG_HOST=127.0.0.1 TG_HTTPS_PORT=0 bash scripts/run_realtime_dual.sh
# 浏览器打开 http://127.0.0.1:8080/
```

`run_realtime_dual.sh` 的默认监听地址是原实验服务器的校园网 IP `192.168.167.5`，默认还会启用 8443 HTTPS。因此在别的机器上请显式设置 `TG_HOST`，并根据是否配置证书设置 `TG_HTTPS_PORT`。物理麦克风通常要求浏览器安全上下文：本机 `localhost` 可直接使用；其他电脑访问需要可信 HTTPS，或把服务器 8080 端口用 SSH 隧道转发为本机 `localhost`。证书和密钥属于部署环境，不在仓库里。双模型服务面向单人操作，模型、视角和音频输入对所有浏览者共享。

网页同时提供：May/Macron 模拟麦克风验证音频、各自离线配音视频、实时帧率与队列信息、视角和动作参数，以及逐帧时延统计。验证音频沿真实在线音频管线运行，不使用预提取特征直接驱动画面；离线视频来自各自检查点的离线验证渲染，需要先按上节生成。

时延大数字追踪**本页面**中一个特征对应音频块的回调，到该帧提交 Canvas 的耗时；可导出每帧原始时间戳和中位数、P95。它不包含麦克风回调前缓冲和屏幕扫描，也不等于物理音画同步误差。其他浏览者没有同一浏览器的音频时间戳，因此不会显示伪造的端到端数值。精确定义、测试与协议见[时延测量文档](realtime/LATENCY_MEASUREMENT.md)。

命令行链路检查（裸 PCM 旧客户端同样兼容）：

```bash
python scripts/test_realtime_file.py \
  --url http://127.0.0.1:8080 \
  --wav data/may/aud.wav --start 220.8 --seconds 8 \
  --output_frame output/may/realtime/file_test_frame.jpg
```

## 5. 仓库内容与引用

| 路径 | 作用 |
| --- | --- |
| `train_mouth.py`、`train_face.py`、`train_fuse.py` | 原有三阶段训练，加入按间隔保存与续训支持 |
| `synthesize_fuse.py`、`synthesize_novel_view.py` | 离线验证和视角渲染 |
| `realtime_server.py`、`realtime/` | 在线服务、DeepSpeech worker、帧追踪与浏览器页面 |
| `scripts/` | 数据检查、训练、音频混流和回归测试脚本 |
| `data/`、`output/` | 本地数据和检查点目录，内容不提交到 GitHub |
| `submodules/` | 两个上游 CUDA 扩展的 Git 子模块 |

论文：Li et al., *TalkingGaussian: Structure-Persistent 3D Talking Head Synthesis via Gaussian Splatting*, ECCV 2024。请引用原论文并保留原作者归属：

```bibtex
@inproceedings{li2024talkinggaussian,
  title={TalkingGaussian: Structure-Persistent 3D Talking Head Synthesis via Gaussian Splatting},
  author={Li, Jiahe and Zhang, Jiawei and Bai, Xiao and Zheng, Jin and Ning, Xin and Zhou, Jun and Gu, Lin},
  booktitle={European Conference on Computer Vision},
  pages={127--145},
  year={2024},
  organization={Springer}
}
```

本仓库延续原作者的研究用途说明；请遵守数据、Basel Face Model、OpenFace、第三方权重及上游子模块各自的使用条件。
