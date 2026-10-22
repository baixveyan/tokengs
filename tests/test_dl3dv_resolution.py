import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
try:
    import pytest
except ModuleNotFoundError:
    if __name__ != "__main__":
        raise
import torch

from tokengs.data.datafield import (
    DF_CAMERA_C2W_TRANSFORM,
    DF_CAMERA_INTRINSICS,
    DF_IMAGE_RGB,
)
from tokengs.data.provider import Provider
from tokengs.data.static.dl3dv import DL3DV10K, DL3DVEval
from tokengs.options import Options


SCENE = "a" * 64
FIELDS = [DF_IMAGE_RGB, DF_CAMERA_C2W_TRANSFORM, DF_CAMERA_INTRINSICS]


def _png(width, height, value):
    buffer = io.BytesIO()
    Image.fromarray(np.full((height, width, 3), value, dtype=np.uint8)).save(
        buffer, format="PNG"
    )
    return buffer.getvalue()


def _write_scene(
    root,
    image_sizes,
    *,
    benchmark=True,
    metadata_size=(1920, 1080),
    missing_metadata_key=None,
    invalid_image_index=None,
):
    root = Path(root)
    archive_root = root if benchmark else root / "1K"
    archive_root.mkdir(parents=True, exist_ok=True)
    prefix = f"{SCENE}/gaussian_splat" if benchmark else SCENE
    image_folder = "images_4" if benchmark else "images"

    frames = []
    members = {}
    for frame_idx, (width, height) in enumerate(image_sizes):
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = frame_idx * 0.1
        image_name = f"frame_{frame_idx:05d}.png"
        frames.append(
            {
                "file_path": f"images/{image_name}",
                "transform_matrix": pose.tolist(),
            }
        )
        members[f"{prefix}/{image_folder}/{image_name}"] = (
            b"not-an-image" if frame_idx == invalid_image_index else _png(width, height, 64 + frame_idx)
        )

    metadata_width, metadata_height = metadata_size
    metadata = {
        "fl_x": 1200.0,
        "fl_y": 1000.0,
        "cx": metadata_width / 2,
        "cy": metadata_height / 2,
        "w": metadata_width,
        "h": metadata_height,
        "frames": frames,
    }
    if missing_metadata_key is not None:
        metadata.pop(missing_metadata_key)

    archive_path = archive_root / f"{SCENE}.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(f"{prefix}/transforms.json", json.dumps(metadata))
        for member, payload in members.items():
            archive.writestr(member, payload)
    return archive_path


def _load_sample(root, actual_size, expected_intrinsics, *, evaluation):
    _write_scene(root, [actual_size, actual_size], benchmark=True)
    if evaluation:
        evaluation_json = Path(root) / "evaluation.json"
        evaluation_json.write_text(
            json.dumps({SCENE: {"context": [0], "target": [1]}}), encoding="utf-8"
        )
        dataset = DL3DVEval(root, evaluation_json, subset=["140"], num_input=1)
    else:
        dataset = DL3DV10K(root, subset=["140"])

    sample = dataset.get_data(0, FIELDS, frame_indices=[0, 1])
    assert tuple(sample[DF_IMAGE_RGB].shape) == (2, 3, actual_size[1], actual_size[0])
    assert tuple(sample[DF_CAMERA_C2W_TRANSFORM].shape) == (2, 4, 4)
    assert tuple(sample[DF_CAMERA_INTRINSICS].shape) == (2, 4)
    torch.testing.assert_close(
        sample[DF_CAMERA_INTRINSICS],
        torch.tensor(expected_intrinsics, dtype=torch.float32).repeat(2, 1),
    )
    torch.testing.assert_close(
        sample[DF_IMAGE_RGB][0, :, 0, 0],
        torch.full((3,), 64 / 255, dtype=torch.float32),
    )


def test_metadata_1920x1080_image_960x540_scales_intrinsics_by_half(tmp_path):
    _load_sample(
        tmp_path,
        (960, 540),
        [600.0, 500.0, 480.0, 270.0],
        evaluation=False,
    )


def test_metadata_1920x1080_image_480x270_scales_intrinsics_by_quarter_for_eval(tmp_path):
    _load_sample(
        tmp_path,
        (480, 270),
        [300.0, 250.0, 240.0, 135.0],
        evaluation=True,
    )


def test_provider_center_crop_and_resize_apply_to_same_intrinsics(tmp_path):
    _write_scene(
        tmp_path,
        [(480, 300)] * 3,
        benchmark=True,
        metadata_size=(1920, 1200),
    )
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(json.dumps([SCENE]), encoding="utf-8")
    options = Options(
        evaluating=True,
        batch_size=1,
        img_size=(256, 448),
        num_input_views=2,
        num_views=3,
        num_workers=0,
        dataset_kwargs={
            "root_path": str(tmp_path),
            "scene_allowlist": str(allowlist),
        },
    )
    provider = Provider("dl3dv_benchmark", options, training=False)
    sample = provider.get_item(0)

    assert tuple(sample["images_all"].shape) == (3, 3, 256, 448)
    crop_height = int(256 * min(300 / 256, 480 / 448))
    shift_y = (crop_height - 300) / 2
    scale_x = 448 / 480
    scale_y = 256 / crop_height
    expected = torch.tensor(
        [
            300.0 * scale_x,
            250.0 * scale_y,
            240.0 * scale_x,
            (150.0 + shift_y) * scale_y,
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(sample["intrinsics_all"], expected.repeat(3, 1))


def test_inconsistent_frame_dimensions_fail_fast_with_context(tmp_path):
    _write_scene(tmp_path, [(960, 540), (480, 270)], benchmark=True)
    dataset = DL3DV10K(tmp_path, subset=["140"])

    with pytest.raises(ValueError, match=rf"Scene {SCENE}, frame 1: inconsistent image size"):
        dataset.get_data(0, FIELDS, frame_indices=[0])


def test_missing_metadata_dimensions_fail_fast_with_scene_context(tmp_path):
    _write_scene(
        tmp_path,
        [(960, 540)],
        benchmark=True,
        missing_metadata_key="h",
    )
    dataset = DL3DV10K(tmp_path, subset=["140"])

    with pytest.raises(ValueError, match=rf"Scene {SCENE}: metadata is missing image dimensions: h"):
        dataset.get_data(0, FIELDS, frame_indices=[0])


def test_invalid_image_fails_fast_with_frame_context(tmp_path):
    _write_scene(
        tmp_path,
        [(960, 540), (960, 540)],
        benchmark=True,
        invalid_image_index=1,
    )
    dataset = DL3DV10K(tmp_path, subset=["140"])

    with pytest.raises(ValueError, match=rf"Scene {SCENE}, frame 1: cannot decode image"):
        dataset.get_data(0, FIELDS, frame_indices=[0])


def test_legacy_training_layout_preserves_native_shape_and_values(tmp_path):
    _write_scene(tmp_path, [(960, 540), (960, 540)], benchmark=False)
    dataset = DL3DV10K(tmp_path, subset=["1K"], resolution="960p_images")
    sample = dataset.get_data(0, FIELDS, frame_indices=[0, 1])

    assert sample["__key__"] == SCENE
    assert tuple(sample[DF_IMAGE_RGB].shape) == (2, 3, 540, 960)
    torch.testing.assert_close(
        sample[DF_CAMERA_INTRINSICS][0],
        torch.tensor([600.0, 500.0, 480.0, 270.0]),
    )


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_inventory(archives):
    rows = []
    for path in archives:
        stat = path.stat()
        rows.append(f"{path.name}|{stat.st_size}|{stat.st_mtime_ns}")
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def _counter_dict(counter):
    return dict(sorted(counter.items()))


def audit_benchmark(root, split_manifest, output):
    root = Path(root).resolve()
    split_manifest = Path(split_manifest).resolve()
    output = Path(output).resolve()
    archives = sorted(root.glob("*.zip"))
    inventory_before = _archive_inventory(archives)
    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    split_by_scene = {
        scene: split
        for split in ("train", "val", "test")
        for scene in manifest[split]
    }

    metadata_scene_sizes = Counter()
    metadata_frame_sizes = Counter()
    image_member_sizes = Counter()
    image_folder_sizes = {}
    loader_scene_sizes = Counter()
    loader_member_sizes = Counter()
    split_counts = Counter()
    anomalies = []
    scenes = []
    total_frames = 0
    total_image_members = 0

    for archive_path in archives:
        scene = archive_path.stem
        split = split_by_scene.get(scene, "unassigned")
        split_counts[split] += 1
        prefix = f"{scene}/gaussian_splat"
        scene_anomalies = []
        per_folder = {}
        referenced_sizes = Counter()
        metadata_size = None
        frame_count = 0

        try:
            with zipfile.ZipFile(archive_path, "r") as archive:
                transform_member = f"{prefix}/transforms.json"
                metadata = json.loads(archive.read(transform_member))
                width = metadata.get("w")
                height = metadata.get("h")
                if (
                    isinstance(width, bool)
                    or isinstance(height, bool)
                    or not isinstance(width, (int, float))
                    or not isinstance(height, (int, float))
                    or width <= 0
                    or height <= 0
                    or int(width) != width
                    or int(height) != height
                ):
                    scene_anomalies.append(
                        {"type": "invalid_metadata_resolution", "w": width, "h": height}
                    )
                else:
                    metadata_size = f"{int(width)}x{int(height)}"
                    metadata_scene_sizes[metadata_size] += 1

                frames = metadata.get("frames")
                if not isinstance(frames, list) or not frames:
                    scene_anomalies.append({"type": "invalid_metadata_frames"})
                    frames = []
                frame_count = len(frames)
                total_frames += frame_count
                if metadata_size is not None:
                    metadata_frame_sizes[metadata_size] += frame_count

                image_members = [
                    name
                    for name in archive.namelist()
                    if not name.endswith("/")
                    and Path(name).suffix.lower() in {".png", ".jpg", ".jpeg"}
                ]
                total_image_members += len(image_members)
                member_sizes = {}
                folder_counters = {}
                for member in image_members:
                    try:
                        with archive.open(member, "r") as stream:
                            with Image.open(stream) as image:
                                width, height = image.size
                        if width <= 0 or height <= 0:
                            raise ValueError(f"non-positive dimensions {width}x{height}")
                        resolution = f"{width}x{height}"
                        member_sizes[member] = resolution
                        image_member_sizes[resolution] += 1
                        folder = Path(member).parent.name
                        folder_counters.setdefault(folder, Counter())[resolution] += 1
                        image_folder_sizes.setdefault(folder, Counter())[resolution] += 1
                    except Exception as exc:
                        scene_anomalies.append(
                            {"type": "invalid_image", "member": member, "error": str(exc)}
                        )

                per_folder = {
                    folder: _counter_dict(counter)
                    for folder, counter in sorted(folder_counters.items())
                }
                for frame_idx, frame in enumerate(frames):
                    file_path = frame.get("file_path") if isinstance(frame, dict) else None
                    if not isinstance(file_path, str) or not file_path:
                        scene_anomalies.append(
                            {"type": "invalid_frame_file_path", "frame": frame_idx}
                        )
                        continue
                    image_name = file_path.replace("\\", "/").rsplit("/", 1)[-1]
                    member = f"{prefix}/images_4/{image_name}"
                    resolution = member_sizes.get(member)
                    if resolution is None:
                        scene_anomalies.append(
                            {"type": "missing_loader_image", "frame": frame_idx, "member": member}
                        )
                    else:
                        referenced_sizes[resolution] += 1

                loader_member_sizes.update(referenced_sizes)
                if len(referenced_sizes) == 1:
                    loader_resolution = next(iter(referenced_sizes))
                    loader_scene_sizes[loader_resolution] += 1
                else:
                    loader_resolution = None
                    if referenced_sizes:
                        scene_anomalies.append(
                            {
                                "type": "inconsistent_loader_image_resolution",
                                "resolutions": _counter_dict(referenced_sizes),
                            }
                        )
        except Exception as exc:
            loader_resolution = None
            scene_anomalies.append({"type": "archive_or_metadata_error", "error": str(exc)})

        for anomaly in scene_anomalies:
            anomalies.append({"scene": scene, "split": split, **anomaly})
        scenes.append(
            {
                "scene": scene,
                "split": split,
                "archive_bytes": archive_path.stat().st_size,
                "metadata_resolution": metadata_size,
                "metadata_frames": frame_count,
                "image_members": sum(sum(values.values()) for values in per_folder.values()),
                "image_resolutions_by_folder": per_folder,
                "loader_images_4_resolution": loader_resolution,
                "loader_images_4_frames_verified": sum(referenced_sizes.values()),
                "anomaly_count": len(scene_anomalies),
            }
        )

    known_480x270 = [
        {"scene": item["scene"], "split": item["split"], "frames": item["metadata_frames"]}
        for item in scenes
        if item["loader_images_4_resolution"] == "480x270"
    ]
    repo = Path(__file__).resolve().parents[1]
    git_head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    inventory_after = _archive_inventory(archives)
    report = {
        "schema_version": 1,
        "audit_scope": "read-only scan of metadata and every PNG/JPEG member in all Benchmark ZIPs",
        "scanned_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(root),
        "scene_count": len(archives),
        "metadata_frame_count": total_frames,
        "image_member_count": total_image_members,
        "split_scene_counts": _counter_dict(split_counts),
        "metadata_resolution_scene_distribution": _counter_dict(metadata_scene_sizes),
        "metadata_resolution_frame_distribution": _counter_dict(metadata_frame_sizes),
        "all_image_member_resolution_distribution": _counter_dict(image_member_sizes),
        "image_resolution_distribution_by_folder": {
            folder: _counter_dict(counter)
            for folder, counter in sorted(image_folder_sizes.items())
        },
        "loader_images_4_scene_resolution_distribution": _counter_dict(loader_scene_sizes),
        "loader_images_4_frame_resolution_distribution": _counter_dict(loader_member_sizes),
        "known_480x270_scenes": known_480x270,
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
        "scenes": scenes,
        "split_manifest": str(split_manifest),
        "code_version": {
            "git_head": git_head,
            "dl3dv_py_sha256": _file_sha256(repo / "tokengs/data/static/dl3dv.py"),
            "test_py_sha256": _file_sha256(Path(__file__)),
        },
        "dataset_inventory": {
            "method": "sha256(sorted archive name|size|mtime_ns)",
            "before": inventory_before,
            "after": inventory_after,
            "unchanged": inventory_before == inventory_after,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit_benchmark(args.audit_root, args.split_manifest, args.output)
    print(
        json.dumps(
            {
                "scene_count": report["scene_count"],
                "metadata_frame_count": report["metadata_frame_count"],
                "image_member_count": report["image_member_count"],
                "known_480x270_scenes": report["known_480x270_scenes"],
                "anomaly_count": report["anomaly_count"],
                "dataset_unchanged": report["dataset_inventory"]["unchanged"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    _main()
