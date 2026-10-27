import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from tokengs.data.datafield import (
    DF_CAMERA_C2W_TRANSFORM,
    DF_CAMERA_INTRINSICS,
    DF_IMAGE_RGB,
    DF_LIDAR_POINTS,
)


def fast_read_ply_vertices(ply_path: Path, stride: int = 4) -> np.ndarray:
    """Fast (~6ms) binary PLY vertex reader that skips parsing the 2.7M triangle face list."""
    with open(ply_path, "rb") as f:
        n_verts = 0
        fmt_ok = False
        in_vertex = False
        vertex_props: List[Tuple[str, str]] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF in PLY header: {ply_path}")
            s = line.decode("ascii", errors="replace").strip()
            if s.startswith("format "):
                fmt_ok = s == "format binary_little_endian 1.0"
            elif s.startswith("element "):
                parts = s.split()
                in_vertex = len(parts) >= 3 and parts[1] == "vertex"
                if in_vertex:
                    n_verts = int(parts[2])
            elif s.startswith("property ") and in_vertex:
                parts = s.split()
                vertex_props.append((parts[1], parts[-1]))
            elif s == "end_header":
                break
        expected_props = [
            ("float", "x"), ("float", "y"), ("float", "z"),
            ("uchar", "red"), ("uchar", "green"), ("uchar", "blue"),
        ]
        if not fmt_ok or n_verts <= 0 or vertex_props != expected_props:
            raise ValueError(
                f"Unsupported PLY header in {ply_path}: format_ok={fmt_ok}, n_verts={n_verts}, props={vertex_props}"
            )
        v_dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
        verts = np.fromfile(f, dtype=v_dtype, count=n_verts)
    return np.stack([verts["x"][::stride], verts["y"][::stride], verts["z"][::stride]], axis=1)


class ScanNetPP:
    """ScanNet++ v2 DSLR dataset loader for TokenGS training and evaluation.

    Expected layout under `root_path`:
      splits/nvs_sem_train.txt, splits/nvs_sem_val.txt
      data/<scene_id>/dslr/nerfstudio/transforms_undistorted.json
      data/<scene_id>/dslr/resized_undistorted_images/*.JPG
      data/<scene_id>/scans/mesh_aligned_0.05.ply
    """

    def __init__(
        self,
        root_path: str,
        split: str = "nvs_sem_train.txt",
        min_valid_frames: int = 24,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.root_path = Path(root_path)
        self.split = split
        self.min_valid_frames = int(min_valid_frames)
        self.is_static = True

        split_file = self.root_path / "splits" / split
        if not split_file.is_file():
            raise FileNotFoundError(f"ScanNet++ split file not found: {split_file}")
        scene_ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.sample_list: List[str] = []
        self._scene_meta_cache: Dict[str, dict] = {}
        for sid in scene_ids:
            scene_dir = self.root_path / "data" / sid
            tj_path = scene_dir / "dslr" / "nerfstudio" / "transforms_undistorted.json"
            img_dir = scene_dir / "dslr" / "resized_undistorted_images"
            if not (tj_path.is_file() and img_dir.is_dir()):
                continue
            meta = self._parse_scene_metadata(sid, tj_path, img_dir)
            if len(meta["frames"]) >= self.min_valid_frames:
                self.sample_list.append(str(scene_dir))
                self._scene_meta_cache[sid] = meta

        if not self.sample_list:
            raise RuntimeError(f"No valid ScanNet++ scenes found for split={split} under {root_path}")

    @staticmethod
    def _parse_scene_metadata(scene_id: str, tj_path: Path, img_dir: Path) -> dict:
        data_dict = json.loads(tj_path.read_text(encoding="utf-8"))
        existing = {p.name for p in img_dir.iterdir()}
        raw_frames = data_dict.get("frames", [])
        if not raw_frames:
            return {"frames": []}
        valid_frames = [
            f for f in raw_frames
            if not f.get("is_bad", False) and f.get("file_path") in existing
        ]
        valid_frames.sort(key=lambda f: f["file_path"])
        if not valid_frames:
            return {"frames": []}

        h, w = int(data_dict["h"]), int(data_dict["w"])
        intrinsics = np.asarray(
            [data_dict["fl_x"], data_dict["fl_y"], data_dict["cx"], data_dict["cy"]],
            dtype=np.float32,
        )
        c2ws = []
        file_paths = []
        for f in valid_frames:
            c2w = np.array(f["transform_matrix"], dtype=np.float32)
            c2w[2, :] *= -1
            c2w = c2w[np.array([1, 0, 2, 3]), :]
            c2w[0:3, 1:3] *= -1
            c2ws.append(c2w)
            file_paths.append(str(img_dir / f["file_path"]))
        return {
            "h": h,
            "w": w,
            "intrinsics": intrinsics,
            "c2ws": np.stack(c2ws, axis=0),
            "frames": file_paths,
        }

    def __len__(self) -> int:
        return len(self.sample_list)

    def count_cameras(self, video_idx: int) -> int:
        return 1

    def count_frames(self, idx: int) -> int:
        sid = Path(self.sample_list[idx]).name
        return len(self._scene_meta_cache[sid]["frames"])

    def get_data(
        self,
        idx: int,
        data_fields: List[str],
        frame_indices: Optional[List[int]] = None,
        view_indices: Optional[List[int]] = None,
        camera_convention: str = "opencv",
        num_depth_frames: Optional[int] = None,
    ) -> dict:
        assert camera_convention == "opencv"
        sid = Path(self.sample_list[idx]).name
        meta = self._scene_meta_cache[sid]
        total_frames = len(meta["frames"])
        if frame_indices is None:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = [int(i) for i in frame_indices]

        img_seq = []
        for f_idx in frame_indices:
            img_path = meta["frames"][f_idx]
            with Image.open(img_path) as im:
                im.load()
                arr = np.array(im.convert("RGB"))
            img_seq.append(arr)

        img_seq_t = torch.from_numpy(np.stack(img_seq, axis=0)).permute(0, 3, 1, 2).contiguous() / 255.0
        c2w_t = torch.from_numpy(meta["c2ws"][frame_indices]).float().contiguous()
        intr = meta["intrinsics"].copy()
        actual_h, actual_w = int(img_seq_t.shape[2]), int(img_seq_t.shape[3])
        if (actual_h, actual_w) != (meta["h"], meta["w"]):
            sx = actual_w / float(meta["w"])
            sy = actual_h / float(meta["h"])
            intr *= np.asarray([sx, sy, sx, sy], dtype=np.float32)
        intr_t = torch.from_numpy(intr).unsqueeze(0).repeat(len(frame_indices), 1)

        out = {"__key__": sid}
        for df in data_fields:
            if df == DF_IMAGE_RGB:
                out[df] = img_seq_t
            elif df == DF_CAMERA_C2W_TRANSFORM:
                out[df] = c2w_t
            elif df == DF_CAMERA_INTRINSICS:
                out[df] = intr_t
            elif df == DF_LIDAR_POINTS:
                mesh_ply = self.root_path / "data" / sid / "scans" / "mesh_aligned_0.05.ply"
                if mesh_ply.is_file():
                    pts_w = fast_read_ply_vertices(mesh_ply, stride=4)
                    out[df] = torch.from_numpy(pts_w).float()
        return out


class ScanNetPPVal(ScanNetPP):
    """Deterministic validation split for ScanNet++ (`nvs_sem_val.txt`, 50 scenes)."""

    def __init__(self, root_path: str, split: str = "nvs_sem_val.txt", num_input_views: int = 6,
                 num_target_views: int = 8, window_size: int = 18, **kwargs):
        super().__init__(root_path=root_path, split=split, min_valid_frames=window_size, **kwargs)
        self.num_input_views = int(num_input_views)
        self.num_target_views = int(num_target_views)
        self.window_size = int(window_size)

    def get_context_target_frames(self, idx: int) -> Tuple[List[int], List[int]]:
        total = self.count_frames(idx)
        win = min(self.window_size, total)
        start = max(0, (total - win) // 3)
        end = start + win - 1
        ctx = np.linspace(start, end, self.num_input_views, dtype=int).tolist()
        ctx_set = set(ctx)
        rem = [f for f in range(start, end + 1) if f not in ctx_set]
        if len(rem) >= self.num_target_views:
            pick = np.linspace(0, len(rem) - 1, self.num_target_views, dtype=int)
            tgt = [rem[i] for i in pick]
        else:
            tgt = np.linspace(start, end, self.num_target_views, dtype=int).tolist()
        return ctx, tgt
