# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import struct
import zipfile
from typing import Dict, List, Optional, Tuple
import os
import numpy as np
import torch
from pathlib import Path
import glob
from PIL import Image
from io import BytesIO

from tokengs.data.datafield import (
    DF_CAMERA_C2W_TRANSFORM,
    DF_CAMERA_INTRINSICS,
    DF_DEPTH,
    DF_IMAGE_RGB,
)


class DL3DV10K:
    def __init__(self, root_path, subset = ['1K', '2K', '3K', '4K', '5K', '6K', '7K', '8K', '9K', '10K', '11K'], resolution = '960p', load_depth=False, scene_allowlist=None, **kwargs):
        """
        data_format: support different formats for what different code base expect
        """
        super().__init__(**kwargs)
        self.root_path = root_path
        self.subset = subset
        self.load_depth = load_depth
        self._image_resolution_cache: Dict[str, Tuple[int, int]] = {}

        self.sample_list = []
        if scene_allowlist is not None:
            # 显式白名单先限定真实场景，再由Provider重复；不扫描全目录。
            with open(scene_allowlist, encoding="utf-8") as handle:
                scenes = json.load(handle)
            if isinstance(scenes, dict):
                scenes = list(scenes)
            if not isinstance(scenes, list) or not scenes:
                raise ValueError("scene_allowlist must be a non-empty JSON list or evaluation dict")
            if any(not isinstance(scene, str) or scene in ("", ".", "..")
                   or "/" in scene or "\\" in scene for scene in scenes):
                raise ValueError("scene_allowlist contains invalid scene names")
            if len(scenes) != len(set(scenes)):
                raise ValueError("scene_allowlist contains duplicate scenes")
            for scene in scenes:
                candidates = [Path(root_path) / f"{scene}.zip" if sub == '140'
                              else Path(root_path) / sub / f"{scene}.zip" for sub in self.subset]
                matches = [path for path in candidates if path.is_file()]
                if not matches:
                    raise FileNotFoundError(f"Missing allowlisted scene ZIP: {scene}")
                if len(matches) != 1:
                    raise ValueError(f"Ambiguous allowlisted scene ZIP: {scene}")
                self.sample_list.append(str(matches[0]))
        else:
            for sub in self.subset:
                if sub == '140':
                    cur_sample_list = sorted(glob.glob(f"{root_path}/*.zip"))
                else:
                    cur_sample_list = sorted(glob.glob(f"{root_path}/{sub}/*.zip"))
                self.sample_list.extend(cur_sample_list)

        if resolution == '960p':
            self.resolution = [540, 960]
            self.image_folder = 'images_4'
        elif resolution == '960p_images':
            # For training dataset that uses 'images' folder instead of 'images_4'
            self.resolution = [540, 960]
            self.image_folder = 'images'
        else:
            raise NotImplementedError(f"Resolution {resolution} not supported")
        
        self.is_static = True

    def __len__(self):
        return len(self.sample_list)
    
    @staticmethod
    def _metadata_resolution(data_dict, scene_name):
        missing = [key for key in ("w", "h") if key not in data_dict]
        if missing:
            raise ValueError(
                f"Scene {scene_name}: metadata is missing image dimensions: {', '.join(missing)}"
            )

        dimensions = {}
        for key in ("w", "h"):
            value = data_dict[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value <= 0
                or int(value) != value
            ):
                raise ValueError(
                    f"Scene {scene_name}: metadata dimension {key} must be a positive integer, got {value!r}"
                )
            dimensions[key] = int(value)
        return dimensions["h"], dimensions["w"]

    @staticmethod
    def _image_member(clip_name, image_folder, frame_data, scene_name, frame_idx):
        if not isinstance(frame_data, dict):
            raise ValueError(f"Scene {scene_name}, frame {frame_idx}: frame metadata must be an object")
        file_path = frame_data.get("file_path")
        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError(f"Scene {scene_name}, frame {frame_idx}: missing valid file_path")
        image_name = file_path.replace("\\", "/").rsplit("/", 1)[-1]
        if image_name in ("", ".", ".."):
            raise ValueError(
                f"Scene {scene_name}, frame {frame_idx}: invalid image file_path {file_path!r}"
            )
        return f"{clip_name}/{image_folder}/{image_name}"

    @staticmethod
    def _read_image_resolution(zip_handle, member, scene_name, frame_idx):
        try:
            with zip_handle.open(member, "r") as stream:
                encoded = stream.read()
            with Image.open(BytesIO(encoded)) as image:
                width, height = image.size
                image.verify()
        except Exception as exc:
            raise ValueError(
                f"Scene {scene_name}, frame {frame_idx}: cannot decode image {member}: {exc}"
            ) from exc
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Scene {scene_name}, frame {frame_idx}: image {member} has invalid size {width}x{height}"
            )
        return height, width

    def _scene_image_resolution(self, zip_path, clip_name, zip_handle, json_data):
        scene_name = Path(zip_path).stem
        self._metadata_resolution(json_data, scene_name)
        frames = json_data.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"Scene {scene_name}: metadata frames must be a non-empty list")

        cache_key = str(Path(zip_path))
        cached = self._image_resolution_cache.get(cache_key)
        if cached is not None:
            return cached

        expected = None
        member_resolutions = {}
        for frame_idx, frame_data in enumerate(frames):
            member = self._image_member(
                clip_name, self.image_folder, frame_data, scene_name, frame_idx
            )
            resolution = member_resolutions.get(member)
            if resolution is None:
                resolution = self._read_image_resolution(
                    zip_handle, member, scene_name, frame_idx
                )
                member_resolutions[member] = resolution
            if expected is None:
                expected = resolution
            elif resolution != expected:
                expected_h, expected_w = expected
                height, width = resolution
                raise ValueError(
                    f"Scene {scene_name}, frame {frame_idx}: inconsistent image size "
                    f"{width}x{height}; expected {expected_w}x{expected_h}"
                )

        self._image_resolution_cache[cache_key] = expected
        return expected

    def load_intrinsics(self, data_dict, resolution=None, scene_name="<unknown>"):
        missing = [key for key in ("fl_x", "fl_y", "cx", "cy") if key not in data_dict]
        if missing:
            raise ValueError(
                f"Scene {scene_name}: metadata is missing intrinsics: {', '.join(missing)}"
            )
        intrinsics = np.asarray(
            [data_dict["fl_x"], data_dict["fl_y"], data_dict["cx"], data_dict["cy"]],
            dtype=np.float32,
        )
        if not np.isfinite(intrinsics).all():
            raise ValueError(f"Scene {scene_name}: camera intrinsics must be finite")

        if resolution is not None:
            height, width = resolution
            if height <= 0 or width <= 0:
                raise ValueError(
                    f"Scene {scene_name}: target image size must be positive, got {width}x{height}"
                )
            metadata_height, metadata_width = self._metadata_resolution(data_dict, scene_name)
            scale_x = width / metadata_width
            scale_y = height / metadata_height
            intrinsics *= np.asarray([scale_x, scale_y, scale_x, scale_y], dtype=np.float32)

        return intrinsics

    def load_video_reader(self, idx):
        zip_path = self.sample_list[idx]
        zip_handle = zipfile.ZipFile(zip_path, "r")

        clip_name = Path(zip_path).stem
        if self.subset == ['140']:
            clip_name = f"{clip_name}/gaussian_splat"

        try:
            with zip_handle.open(f"{clip_name}/transforms.json", "r") as f:
                json_data = json.load(f)
            image_resolution = self._scene_image_resolution(
                zip_path, clip_name, zip_handle, json_data
            )
            video_length = len(json_data["frames"])
            intrinsics = self.load_intrinsics(
                json_data,
                resolution=image_resolution,
                scene_name=Path(zip_path).stem,
            )
            transform_matrix_all = self.load_cameras(json_data)
        except Exception:
            zip_handle.close()
            raise

        return clip_name, video_length, intrinsics, transform_matrix_all, zip_handle, json_data

    def load_cameras(self, data_dict):
        transform_matrix_all = []
        for frame_data in data_dict['frames']:
            transform_matrix = np.array(frame_data['transform_matrix'])
            c2w = transform_matrix
            c2w[2, :] *= -1
            c2w = c2w[np.array([1, 0, 2, 3]), :]
            c2w[0:3, 1:3] *= -1
            transform_matrix_all.append(c2w)
        return np.stack(transform_matrix_all, axis=0)

    def count_cameras(self, video_idx: int) -> int:
        return 1
    
    def count_frames(self, idx):
        zip_path = self.sample_list[idx]
        with zipfile.ZipFile(zip_path, "r") as zip_handle:
            clip_name = Path(zip_path).stem
            if self.subset == ['140']:
                clip_name = f"{clip_name}/gaussian_splat"
            # load json data
            with zip_handle.open(f"{clip_name}/transforms.json", "r") as f:
                json_data = json.load(f)
            total_frames = len(json_data['frames'])

        return total_frames

    def get_data(
        self,
        idx,
        data_fields: List[str],
        frame_indices: Optional[List[int]] = None,
        view_indices: List[int] = None,
        camera_convention: str = "opencv",
        num_depth_frames: Optional[int] = None,
    ):
        assert camera_convention == "opencv"

        clip_name, total_frames, intrinsics, transform_matrices, zip_handle, json_data = self.load_video_reader(idx)
        scene_name = Path(self.sample_list[idx]).stem
        image_resolution = self._image_resolution_cache[str(Path(self.sample_list[idx]))]
        if frame_indices is None:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = [int(frame_idx) for frame_idx in frame_indices]
        if not frame_indices:
            zip_handle.close()
            raise ValueError(f"Scene {scene_name}: frame_indices must not be empty")

        try:
            for frame_idx in frame_indices:
                if frame_idx < 0 or frame_idx >= total_frames:
                    raise IndexError(
                        f"Scene {scene_name}, frame {frame_idx}: index outside [0, {total_frames})"
                    )
            c2w = transform_matrices[frame_indices]

            img_seq = []
            for frame_idx in frame_indices:
                member = self._image_member(
                    clip_name,
                    self.image_folder,
                    json_data["frames"][frame_idx],
                    scene_name,
                    frame_idx,
                )
                try:
                    with zip_handle.open(member, "r") as stream:
                        encoded = stream.read()
                    with Image.open(BytesIO(encoded)) as image:
                        image.load()
                        decoded_resolution = (image.height, image.width)
                        image_array = np.array(image.convert("RGB"))
                except Exception as exc:
                    raise ValueError(
                        f"Scene {scene_name}, frame {frame_idx}: cannot decode image {member}: {exc}"
                    ) from exc
                if decoded_resolution != image_resolution:
                    expected_h, expected_w = image_resolution
                    height, width = decoded_resolution
                    raise ValueError(
                        f"Scene {scene_name}, frame {frame_idx}: decoded image size "
                        f"{width}x{height}; expected {expected_w}x{expected_h}"
                    )
                img_seq.append(image_array)

            img_seq = np.stack(img_seq, axis=0)  # n h w c
            img_seq = torch.from_numpy(img_seq).permute(0, 3, 1, 2).contiguous()
            img_seq = img_seq / 255.0  # (0,1)

            c2w = torch.from_numpy(c2w).float().contiguous()
            intrinsics = torch.from_numpy(intrinsics).unsqueeze(0).repeat(len(frame_indices), 1)

            output_dict = {"__key__": clip_name}
            for data_field in data_fields:
                if data_field == DF_IMAGE_RGB:
                    output_dict[data_field] = img_seq
                elif data_field == DF_CAMERA_C2W_TRANSFORM:
                    output_dict[data_field] = c2w
                elif data_field == DF_CAMERA_INTRINSICS:
                    output_dict[data_field] = intrinsics
                elif data_field == DF_DEPTH and self.load_depth:
                    depth_seq = self._load_depth_seq(
                        clip_name,
                        json_data,
                        zip_handle,
                        frame_indices,
                        num_depth_frames,
                        image_resolution=image_resolution,
                        intrinsics=intrinsics[0].numpy(),
                    )
                    output_dict[data_field] = torch.from_numpy(depth_seq).float().unsqueeze(1)
            return output_dict
        finally:
            zip_handle.close()

    def _load_depth_seq(
        self,
        clip_name,
        json_data,
        zip_handle,
        frame_indices,
        num_depth_frames: Optional[int],
        image_resolution=None,
        intrinsics=None,
    ) -> np.ndarray:
        n_load = len(frame_indices) if num_depth_frames is None else min(num_depth_frames, len(frame_indices))
        depth_seq = []
        for i, frame_idx in enumerate(frame_indices):
            if i < n_load:
                depth_path = f"{clip_name}/{json_data['frames'][frame_idx]['depth_path']}"
                with zip_handle.open(depth_path, "r") as f:
                    depth = np.load(BytesIO(f.read()))
            else:
                resolution = self.resolution if image_resolution is None else image_resolution
                depth = np.zeros(resolution, dtype=np.float64)
            depth_seq.append(depth)
        return np.stack(depth_seq, axis=0)


class DL3DVEval(DL3DV10K):
    def __init__(self, root_path, evaluation_json, subset = ['1K', '2K', '3K', '4K', '5K', '6K', '7K', '8K', '9K', '10K', '11K'], resolution = '960p', num_input=16, load_depth=False):
        super().__init__(root_path, subset, resolution, load_depth=load_depth)

        self.evaluation_indices = json.load(open(evaluation_json, "r"))
        self.sample_list = []
        for k in self.evaluation_indices:
            if isinstance(k, str):
                self.sample_list.append(os.path.join(root_path, k+'.zip'))
            else:
                self.sample_list.append(os.path.join(root_path, k['scene_name']+'.zip'))
        
        self.is_static = True

        self.num_input = num_input
        self._colmap_points_cache: Dict[str, Tuple[np.ndarray, List[frozenset]]] = {}

    def _read_points3d_bin(
        self, zip_handle: zipfile.ZipFile, clip_name: str
    ) -> Tuple[np.ndarray, List[frozenset]]:
        path = f"{clip_name}/sparse/0/points3D.bin"
        with zip_handle.open(path, "r") as f:
            data = f.read()

        off = 0
        (n_points,) = struct.unpack_from("<Q", data, off)
        off += 8
        xyz = np.empty((n_points, 3), dtype=np.float64)
        image_id_sets: List[frozenset] = []
        for i in range(n_points):
            off += 8
            xyz[i] = struct.unpack_from("<ddd", data, off)
            off += 24 + 3 + 8
            (track_len,) = struct.unpack_from("<Q", data, off)
            off += 8
            track = struct.unpack_from("<" + "II" * track_len, data, off)
            off += 8 * track_len
            image_id_sets.append(frozenset(track[::2]))
        return xyz, image_id_sets

    def _get_colmap_points(
        self, zip_handle: zipfile.ZipFile, clip_name: str
    ) -> Tuple[np.ndarray, List[frozenset]]:
        cached = self._colmap_points_cache.get(clip_name)
        if cached is not None:
            return cached
        cached = self._read_points3d_bin(zip_handle, clip_name)
        self._colmap_points_cache[clip_name] = cached
        return cached

    @staticmethod
    def _colmap_world_to_final_world_transform(json_data: Dict) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        applied = json_data.get("applied_transform")
        if applied is not None:
            transform[:3, :] = np.asarray(applied, dtype=np.float64)

        z_flip = np.diag([1.0, 1.0, -1.0, 1.0])
        xy_swap = np.array(
            [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        )
        return xy_swap @ z_flip @ transform

    def _load_depth_seq(
        self,
        clip_name,
        json_data,
        zip_handle,
        frame_indices,
        num_depth_frames: Optional[int],
        image_resolution=None,
        intrinsics=None,
    ) -> np.ndarray:
        xyz_all, image_id_sets = self._get_colmap_points(zip_handle, clip_name)
        world_transform = self._colmap_world_to_final_world_transform(json_data)

        if image_resolution is None:
            image_resolution = tuple(self.resolution)
        if intrinsics is None:
            intrinsics = self.load_intrinsics(
                json_data, resolution=image_resolution, scene_name=clip_name
            )
        fx, fy, cx, cy = intrinsics
        c2ws_all = self.load_cameras(json_data)
        H, W = image_resolution

        xyz_h = np.concatenate(
            [xyz_all, np.ones((xyz_all.shape[0], 1), dtype=np.float64)], axis=1
        )
        xyz_world = (xyz_h @ world_transform.T)[:, :3]

        n_frames = len(frame_indices)
        n_load = n_frames if num_depth_frames is None else min(num_depth_frames, n_frames)
        depth_seq = np.zeros((n_frames, H, W), dtype=np.float64)
        for i in range(n_load):
            frame_idx = int(frame_indices[i])
            colmap_image_id = json_data["frames"][frame_idx].get("colmap_im_id")
            if colmap_image_id is None:
                continue

            mask = np.fromiter(
                (colmap_image_id in s for s in image_id_sets),
                dtype=bool,
                count=len(image_id_sets),
            )
            if not mask.any():
                continue

            c2w = c2ws_all[frame_idx]
            w2c = np.linalg.inv(c2w)
            pts_cam = xyz_world[mask] @ w2c[:3, :3].T + w2c[:3, 3]
            z = pts_cam[:, 2]
            in_front = z > 1e-6
            if not in_front.any():
                continue
            pts_cam = pts_cam[in_front]
            z = z[in_front]

            u = fx * pts_cam[:, 0] / z + cx
            v = fy * pts_cam[:, 1] / z + cy
            u_i = np.round(u).astype(np.int64)
            v_i = np.round(v).astype(np.int64)
            in_bounds = (u_i >= 0) & (u_i < W) & (v_i >= 0) & (v_i < H)
            if not in_bounds.any():
                continue

            u_i = u_i[in_bounds]
            v_i = v_i[in_bounds]
            z = z[in_bounds]
            order = np.argsort(-z)
            depth_seq[i, v_i[order], u_i[order]] = z[order]

        return depth_seq

    def get_context_target_frames(self, idx):
        if isinstance(self.evaluation_indices, dict):
            scene_name = Path(self.sample_list[idx]).stem
            eval_data = self.evaluation_indices[scene_name]
            context_frames = eval_data["context"]
            target_frames = eval_data["target"]
            return context_frames, target_frames
        else:
            # llrm eval
            eval_data = self.evaluation_indices[idx]
            if self.num_input == 16:
                context_frames = eval_data["fold_8_kmeans_16_input"]
            elif self.num_input == 32:
                context_frames = eval_data["fold_8_kmeans_32_input"]
            else:
                raise ValueError(f"Unsupported number of input frames: {self.num_input}")
            target_frames = [x for x in range(self.count_frames(idx))]
            return context_frames, target_frames
