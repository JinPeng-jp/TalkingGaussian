"""Persistent one-frame-at-a-time renderer for the May TalkingGaussian avatar."""

import copy
import math
import os
import threading
import time

import cv2
import numpy as np
import torch

from gaussian_renderer import (
    GaussianModel,
    MotionNetwork,
    MouthMotionNetwork,
    render_motion,
    render_motion_mouth,
)
from scene import Scene


def _rotation_matrix(yaw_deg, pitch_deg, roll_deg):
    yaw, pitch, roll = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    cy, sy = math.cos(yaw), math.sin(yaw)
    cx, sx = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(roll), math.sin(roll)
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rot_z @ rot_x @ rot_y


def _blend_camera_pose(base_c2w, target_c2w, weight):
    """Blend two camera poses and project the rotation back onto SO(3)."""
    weight = float(np.clip(weight, 0.0, 1.0))
    if weight <= 0.0:
        return base_c2w
    blended = np.eye(4, dtype=np.float32)
    raw_rotation = (
        (1.0 - weight) * base_c2w[:3, :3]
        + weight * target_c2w[:3, :3]
    )
    u, _, vh = np.linalg.svd(raw_rotation)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    blended[:3, :3] = rotation.astype(np.float32)
    blended[:3, 3] = (
        (1.0 - weight) * base_c2w[:3, 3]
        + weight * target_c2w[:3, 3]
    )
    return blended


class RealtimeAvatarRenderer:
    def __init__(
        self,
        dataset,
        pipeline,
        render_gpu=0,
        camera_index=0,
        jpeg_quality=88,
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        zoom=1.0,
    ):
        torch.cuda.set_device(int(render_gpu))
        self.render_gpu = int(render_gpu)
        self.pipeline = pipeline
        self.jpeg_quality = int(jpeg_quality)
        self.control_lock = threading.Lock()
        self.control = {
            "yaw": float(yaw),
            "pitch": float(pitch),
            "roll": float(roll),
            "zoom": float(zoom),
            "expression_intensity": 1.0,
        }

        # The validation split supplies the camera, background and expression
        # metadata required by the released model.
        dataset.eval = True
        self.dataset = dataset
        self.gaussians = GaussianModel(dataset.sh_degree)
        self.gaussians_mouth = GaussianModel(dataset.sh_degree)
        self.scene = Scene(dataset, self.gaussians, shuffle=False)
        self.views = self.scene.getTestCameras()
        if not self.views:
            raise RuntimeError("No validation cameras were loaded")
        if not 0 <= camera_index < len(self.views):
            raise ValueError(
                "camera_index {} is outside [0, {})".format(
                    camera_index, len(self.views)
                )
            )

        self.motion_net = MotionNetwork(args=dataset).cuda().eval()
        self.motion_net_mouth = MouthMotionNetwork(args=dataset).cuda().eval()
        checkpoint_path = os.path.join(dataset.model_path, "chkpnt_fuse_latest.pth")
        checkpoint = torch.load(checkpoint_path)
        if len(checkpoint) != 4:
            raise ValueError("Realtime inference requires the four-part fuse checkpoint")
        model_params, motion_params, mouth_params, mouth_motion_params = checkpoint
        self.gaussians.restore(model_params, None)
        self.motion_net.load_state_dict(motion_params, strict=False)
        self.gaussians_mouth.restore(mouth_params, None)
        self.motion_net_mouth.load_state_dict(mouth_motion_params, strict=False)

        self.camera = copy.deepcopy(self.views[camera_index])
        self.camera.background = self.camera.background.cuda()
        base_w2c = (
            self.camera.world_view_transform.transpose(0, 1)
            .detach()
            .cpu()
            .numpy()
        )
        self.base_c2w = np.linalg.inv(base_w2c).astype(np.float32)
        self.motion_c2ws = []
        for view in self.views:
            view_w2c = (
                view.world_view_transform.transpose(0, 1)
                .detach()
                .cpu()
                .numpy()
            )
            self.motion_c2ws.append(np.linalg.inv(view_w2c).astype(np.float32))
        self.orbit_center = np.zeros(3, dtype=np.float32)
        self.background = torch.tensor(
            [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
            dtype=torch.float32,
            device="cuda",
        )
        # Online audio has no matching Action Unit stream.  Cycling validation
        # AUs here would create unrelated eyebrow/blink motion even while the
        # microphone is stopped, so keep one expression fixed.  Audio remains
        # the only time-varying driver in realtime mode.
        self.fixed_expression = (
            self.views[camera_index].talking_dict["au_exp"].detach().clone().cuda()
        )
        self.motion_expressions = [
            view.talking_dict["au_exp"].detach().cuda() for view in self.views
        ]
        self.motion_index = 0
        self.motion_direction = 1
        self.motion_last_at = None
        self.motion_fps = 25.0
        self.motion_lock = threading.Lock()
        self.neutral_audio = np.zeros((8, 29, 16), dtype=np.float32)
        self._last_audio_source = None
        self._last_audio_tensor = None
        self._last_camera_key = None

    def set_control(
        self,
        yaw=None,
        pitch=None,
        roll=None,
        zoom=None,
        expression_intensity=None,
    ):
        with self.control_lock:
            updates = {
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
                "zoom": zoom,
                "expression_intensity": expression_intensity,
            }
            for name, value in updates.items():
                if value is not None:
                    self.control[name] = float(value)
            self.control["yaw"] = float(np.clip(self.control["yaw"], -25.0, 25.0))
            self.control["pitch"] = float(
                np.clip(self.control["pitch"], -15.0, 15.0)
            )
            self.control["roll"] = float(np.clip(self.control["roll"], -20.0, 20.0))
            self.control["zoom"] = float(np.clip(self.control["zoom"], 0.75, 1.35))
            self.control["expression_intensity"] = float(
                np.clip(self.control["expression_intensity"], 0.0, 1.5)
            )
            return dict(self.control)

    def get_control(self):
        with self.control_lock:
            return dict(self.control)

    def _update_camera(self, source_c2w=None, control=None):
        if control is None:
            control = self.get_control()
        if source_c2w is None:
            source_c2w = self.base_c2w
        camera_key = (
            control["yaw"], control["pitch"], control["roll"], control["zoom"],
            source_c2w.tobytes(),
        )
        if camera_key == self._last_camera_key:
            return control
        orbit = _rotation_matrix(
            control["yaw"], control["pitch"], control["roll"]
        )
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = orbit @ source_c2w[:3, :3]
        relative_center = source_c2w[:3, 3] - self.orbit_center
        c2w[:3, 3] = self.orbit_center + control["zoom"] * (
            orbit @ relative_center
        )
        w2c = np.linalg.inv(c2w).astype(np.float32)
        world_view = torch.from_numpy(w2c).transpose(0, 1).cuda()
        self.camera.world_view_transform = world_view
        self.camera.full_proj_transform = (
            world_view.unsqueeze(0).bmm(self.camera.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        # world_view = inverse(c2w).T, so its inverse's last row is c2w's
        # translation.  Avoid a redundant GPU matrix inverse every frame.
        self.camera.camera_center = torch.from_numpy(
            np.ascontiguousarray(c2w[:3, 3])
        ).cuda()
        self._last_camera_key = camera_key
        return control

    def _next_motion_index(self):
        with self.motion_lock:
            now = time.perf_counter()
            if self.motion_last_at is None:
                self.motion_last_at = now
            else:
                steps = int((now - self.motion_last_at) * self.motion_fps)
                for _ in range(steps):
                    if len(self.views) <= 1:
                        break
                    next_index = self.motion_index + self.motion_direction
                    if next_index >= len(self.views) or next_index < 0:
                        self.motion_direction *= -1
                        next_index = self.motion_index + self.motion_direction
                    self.motion_index = next_index
                if steps:
                    self.motion_last_at += steps / self.motion_fps
            index = self.motion_index
            return index

    def reset_motion(self):
        with self.motion_lock:
            self.motion_index = 0
            self.motion_direction = 1
            self.motion_last_at = None

    def render_rgb(self, audio_features=None, motion_strength=0.0):
        if audio_features is None:
            audio_features = self.neutral_audio
        if audio_features is not self._last_audio_source:
            audio_tensor_source = np.asarray(audio_features, dtype=np.float32)
            if audio_tensor_source.shape != (8, 29, 16):
                raise ValueError(
                    "Expected realtime audio features with shape (8, 29, 16), got {}".format(
                        audio_tensor_source.shape
                    )
                )
            if not np.isfinite(audio_tensor_source).all():
                audio_tensor_source = np.nan_to_num(audio_tensor_source).astype(np.float32)
            self._last_audio_tensor = torch.from_numpy(
                np.ascontiguousarray(audio_tensor_source)
            ).cuda()
            self._last_audio_source = audio_features

        control = self.get_control()
        natural_strength = float(
            np.clip(
                motion_strength * control["expression_intensity"],
                0.0,
                1.0,
            )
        )
        source_c2w = self.base_c2w
        expression = self.fixed_expression
        if natural_strength > 1e-3:
            driver_index = self._next_motion_index()
            expression = self.fixed_expression + natural_strength * (
                self.motion_expressions[driver_index] - self.fixed_expression
            )
            # Pose motion is deliberately weaker than AU motion because the
            # avatar was trained from a monocular sequence.
            source_c2w = _blend_camera_pose(
                self.base_c2w,
                self.motion_c2ws[driver_index],
                natural_strength * 0.35,
            )
        else:
            # An idle period must not fast-forward the 25 Hz source motion.
            with self.motion_lock:
                self.motion_last_at = None

        self.camera.talking_dict["auds"] = self._last_audio_tensor
        self.camera.talking_dict["au_exp"] = expression
        self._update_camera(source_c2w=source_c2w, control=control)

        with torch.no_grad():
            face = render_motion(
                self.camera,
                self.gaussians,
                self.motion_net,
                self.pipeline,
                self.background,
                frame_idx=0,
            )
            mouth = render_motion_mouth(
                self.camera,
                self.gaussians_mouth,
                self.motion_net_mouth,
                self.pipeline,
                self.background,
                frame_idx=0,
            )
            mouth_image = mouth["render"] + self.camera.background / 255.0 * (
                1.0 - mouth["alpha"]
            )
            image = face["render"] + mouth_image * (1.0 - face["alpha"])
            image = image[:3].clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
        return np.ascontiguousarray((image * 255).astype(np.uint8))

    def render_jpeg(self, audio_features=None, motion_strength=0.0):
        started = time.perf_counter()
        rgb = self.render_rgb(audio_features, motion_strength=motion_strength)
        ok, encoded = cv2.imencode(
            ".jpg",
            rgb[:, :, ::-1],
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("OpenCV failed to encode a realtime JPEG frame")
        return encoded.tobytes(), (time.perf_counter() - started) * 1000.0

    def warm_up(self, frames=3):
        timings = []
        for _ in range(int(frames)):
            _, elapsed = self.render_jpeg(self.neutral_audio)
            timings.append(elapsed)
        torch.cuda.synchronize()
        return timings
