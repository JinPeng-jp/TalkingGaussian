"""Render TalkingGaussian from controllable novel viewpoints.

The original ``synthesize_fuse.py`` only renders the camera poses stored in the
dataset.  This entry point keeps a single May background/torso frame fixed,
drives expression and mouth motion with the validation sequence, and orbits the
virtual camera around the Gaussian head.
"""

import copy
import math
import os
import subprocess
from argparse import ArgumentParser

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import (
    GaussianModel,
    MotionNetwork,
    MouthMotionNetwork,
    render_motion,
    render_motion_mouth,
)
from scene import Scene
from utils.general_utils import safe_state


def _rotation_matrix(yaw_deg, pitch_deg, roll_deg):
    """Return a world-space camera-orbit rotation (Y yaw, X pitch, Z roll)."""
    yaw, pitch, roll = np.deg2rad([yaw_deg, pitch_deg, roll_deg])

    cy, sy = math.cos(yaw), math.sin(yaw)
    cx, sx = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(roll), math.sin(roll)

    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rot_z @ rot_x @ rot_y


def _set_orbit(camera, base_c2w, yaw, pitch, roll, orbit_center, zoom):
    """Orbit a camera around ``orbit_center`` while preserving its intrinsics."""
    orbit = _rotation_matrix(yaw, pitch, roll)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = orbit @ base_c2w[:3, :3]
    relative_center = base_c2w[:3, 3] - orbit_center
    c2w[:3, 3] = orbit_center + zoom * (orbit @ relative_center)

    w2c = np.linalg.inv(c2w).astype(np.float32)
    world_view = torch.from_numpy(w2c).transpose(0, 1).cuda()
    camera.world_view_transform = world_view
    camera.full_proj_transform = (
        world_view.unsqueeze(0).bmm(camera.projection_matrix.unsqueeze(0))
    ).squeeze(0)
    camera.camera_center = world_view.inverse()[3, :3]


def _parse_float_list(value):
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one yaw angle is required")
    return values


def _view_angles(args, frame_idx, num_frames):
    if args.view_mode == "fixed":
        yaw = args.yaw
    elif args.view_mode == "switch":
        sequence = _parse_float_list(args.yaw_sequence)
        yaw = sequence[(frame_idx // args.hold_frames) % len(sequence)]
    elif args.view_mode == "sweep":
        # Start and end at the frontal view and smoothly visit both sides.
        phase = 2.0 * math.pi * frame_idx / max(1, num_frames - 1)
        yaw = args.yaw + args.yaw_amplitude * math.sin(phase)
    else:
        raise ValueError("Unknown view mode: {}".format(args.view_mode))
    return yaw, args.pitch, args.roll


def _compose(view, face_package, mouth_package, dilate):
    if dilate:
        alpha_mouth = F.max_pool2d(
            mouth_package["alpha"][None], kernel_size=13, stride=1, padding=6
        )[0]
    else:
        alpha_mouth = mouth_package["alpha"]

    mouth_image = (
        mouth_package["render"]
        + view.background.cuda() / 255.0 * (1.0 - alpha_mouth)
    )
    image = face_package["render"] + mouth_image * (1.0 - face_package["alpha"])
    image = image[:3].clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
    # Torch's HWC permutation may retain non-contiguous strides.  ImageIO can
    # consume it, but OpenCV's optional text overlay requires C-contiguous data.
    return np.ascontiguousarray((image * 255).astype(np.uint8))


def _mux_audio(video_path, output_path, wav_path, start_seconds, duration):
    environment = os.environ.copy()
    environment.pop("LD_LIBRARY_PATH", None)
    command = [
        "/usr/bin/ffmpeg",
        "-nostdin",
        "-y",
        "-v",
        "warning",
        "-i",
        video_path,
        "-ss",
        str(start_seconds),
        "-i",
        wav_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "64k",
        "-t",
        str(duration),
        output_path,
    ]
    subprocess.run(command, check=True, env=environment)


def render_novel_views(dataset, pipeline, args):
    # Novel-view rendering only needs the compact validation camera set.
    dataset.eval = True

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        gaussians_mouth = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False)

        motion_net = MotionNetwork(args=dataset).cuda().eval()
        motion_net_mouth = MouthMotionNetwork(args=dataset).cuda().eval()
        checkpoint_path = os.path.join(dataset.model_path, "chkpnt_fuse_latest.pth")
        checkpoint = torch.load(checkpoint_path)
        if len(checkpoint) != 4:
            raise ValueError(
                "Expected the four-part inference checkpoint at {}, got {} entries".format(
                    checkpoint_path, len(checkpoint)
                )
            )
        model_params, motion_params, mouth_params, mouth_motion_params = checkpoint
        gaussians.restore(model_params, None)
        motion_net.load_state_dict(motion_params, strict=False)
        gaussians_mouth.restore(mouth_params, None)
        motion_net_mouth.load_state_dict(mouth_motion_params, strict=False)

        views = scene.getTestCameras()
        if not views:
            raise RuntimeError("No validation cameras were loaded")
        if not 0 <= args.camera_index < len(views):
            raise ValueError(
                "camera_index {} is outside [0, {})".format(args.camera_index, len(views))
            )

        num_frames = args.num_frames if args.num_frames > 0 else len(views)
        if num_frames > len(views):
            raise ValueError(
                "num_frames={} exceeds the {} loaded audio/expression frames".format(
                    num_frames, len(views)
                )
            )

        template = copy.deepcopy(views[args.camera_index])
        template.background = template.background.cuda()
        base_w2c = template.world_view_transform.transpose(0, 1).detach().cpu().numpy()
        base_c2w = np.linalg.inv(base_w2c).astype(np.float32)
        orbit_center = np.asarray(_parse_float_list(args.orbit_center), dtype=np.float32)
        if orbit_center.shape != (3,):
            raise ValueError("orbit_center must contain exactly three comma-separated values")

        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        writer = imageio.get_writer(
            args.output,
            fps=args.fps,
            quality=8,
            macro_block_size=1,
        )
        bg_color = [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        try:
            for frame_idx in tqdm(range(num_frames), desc="Novel-view rendering", ascii=True):
                driver = views[frame_idx]
                template.talking_dict["auds"] = driver.talking_dict["auds"]
                template.talking_dict["au_exp"] = driver.talking_dict["au_exp"]

                yaw, pitch, roll = _view_angles(args, frame_idx, num_frames)
                _set_orbit(
                    template,
                    base_c2w,
                    yaw,
                    pitch,
                    roll,
                    orbit_center,
                    args.zoom,
                )

                face = render_motion(
                    template, gaussians, motion_net, pipeline, background, frame_idx=0
                )
                mouth = render_motion_mouth(
                    template,
                    gaussians_mouth,
                    motion_net_mouth,
                    pipeline,
                    background,
                    frame_idx=0,
                )
                frame = _compose(template, face, mouth, args.dilate)
                if args.draw_label:
                    cv2.putText(
                        frame,
                        "yaw {:+.1f}  pitch {:+.1f}".format(yaw, pitch),
                        (14, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        "yaw {:+.1f}  pitch {:+.1f}".format(yaw, pitch),
                        (14, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (25, 25, 25),
                        1,
                        cv2.LINE_AA,
                    )
                writer.append_data(frame)
        finally:
            writer.close()

    print("[NOVEL_VIEW_OUTPUT] {}".format(args.output))
    if args.with_audio:
        wav_path = args.audio_wav or os.path.join(dataset.source_path, "aud.wav")
        if not os.path.isfile(wav_path):
            raise FileNotFoundError("Audio file not found: {}".format(wav_path))
        if args.audio_start >= 0:
            audio_start = args.audio_start
        elif dataset.audio:
            audio_start = 0.0
        else:
            audio_start = float(views[0].talking_dict["img_id"]) / args.fps
        stem, extension = os.path.splitext(args.output)
        audio_output = stem + "_with_audio" + (extension or ".mp4")
        _mux_audio(
            args.output,
            audio_output,
            wav_path,
            audio_start,
            num_frames / args.fps,
        )
        print("[NOVEL_VIEW_AUDIO_OUTPUT] {}".format(audio_output))


if __name__ == "__main__":
    parser = ArgumentParser(description="TalkingGaussian novel-view renderer")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--view_mode", choices=["fixed", "switch", "sweep"], default="fixed")
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--roll", type=float, default=0.0)
    parser.add_argument("--yaw_sequence", type=str, default="-15,-10,-5,0,5,10,15")
    parser.add_argument("--yaw_amplitude", type=float, default=15.0)
    parser.add_argument("--hold_frames", type=int, default=25)
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--zoom", type=float, default=1.0)
    parser.add_argument("--orbit_center", type=str, default="0,0,0")
    parser.add_argument("--dilate", action="store_true")
    parser.add_argument("--draw_label", action="store_true")
    parser.add_argument(
        "--output",
        type=str,
        default="output/may/novel_view/novel_view.mp4",
    )
    parser.add_argument("--with_audio", action="store_true")
    parser.add_argument("--audio_wav", type=str, default="")
    parser.add_argument("--audio_start", type=float, default=-1.0)
    args = get_combined_args(parser)

    if args.hold_frames <= 0:
        raise ValueError("hold_frames must be positive")
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if args.zoom <= 0:
        raise ValueError("zoom must be positive")

    safe_state(args.quiet)
    render_novel_views(model.extract(args), pipeline.extract(args), args)
