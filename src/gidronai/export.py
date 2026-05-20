"""Multi-format data export pipeline for training datasets."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from gidronai.config import ExportConfig
from gidronai.engine import SceneGraph, SceneNode, NodeType

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox2D:
    """Axis-aligned 2D bounding box in pixel coordinates."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int
    label: str
    instance_id: int

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_coco(self) -> dict[str, Any]:
        return {
            "bbox": [self.x_min, self.y_min, self.width, self.height],
            "category_id": hash(self.label) % 1000,
            "area": self.area,
            "iscrowd": 0,
        }

    def to_kitti(self) -> str:
        return (
            f"{self.label} 0.00 0 0.00 "
            f"{self.x_min:.2f} {self.y_min:.2f} {self.x_max:.2f} {self.y_max:.2f} "
            f"0.00 0.00 0.00 0.00 0.00 0.00 0.00"
        )


@dataclass
class FrameData:
    """Captured data for a single simulation frame."""

    frame_id: int
    timestamp: float
    rgb: np.ndarray | None = None
    depth: np.ndarray | None = None
    segmentation: np.ndarray | None = None
    bounding_boxes: list[BoundingBox2D] = field(default_factory=list)
    agent_positions: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class _COCOWriter:
    """Writes annotations in COCO JSON format."""

    def __init__(self, output_dir: Path, resolution: tuple[int, int]) -> None:
        self.output_dir = output_dir
        self.resolution = resolution
        self._images: list[dict[str, Any]] = []
        self._annotations: list[dict[str, Any]] = []
        self._categories: dict[str, int] = {}
        self._ann_id = 0

    def add_frame(self, frame: FrameData) -> None:
        w, h = self.resolution
        self._images.append({
            "id": frame.frame_id,
            "file_name": f"frame_{frame.frame_id:06d}.png",
            "width": w,
            "height": h,
        })

        for bbox in frame.bounding_boxes:
            if bbox.label not in self._categories:
                self._categories[bbox.label] = len(self._categories) + 1

            self._ann_id += 1
            ann = bbox.to_coco()
            ann["id"] = self._ann_id
            ann["image_id"] = frame.frame_id
            ann["category_id"] = self._categories[bbox.label]
            self._annotations.append(ann)

    def finalize(self) -> None:
        categories = [
            {"id": cid, "name": name} for name, cid in self._categories.items()
        ]
        dataset = {
            "images": self._images,
            "annotations": self._annotations,
            "categories": categories,
        }
        out_path = self.output_dir / "annotations.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)
        logger.info("COCO annotations written: %d images, %d annotations", len(self._images), len(self._annotations))


class _KITTIWriter:
    """Writes annotations in KITTI txt format."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self._label_dir = output_dir / "label_2"
        self._label_dir.mkdir(parents=True, exist_ok=True)

    def add_frame(self, frame: FrameData) -> None:
        lines = [bbox.to_kitti() for bbox in frame.bounding_boxes]
        out_path = self._label_dir / f"{frame.frame_id:06d}.txt"
        with out_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def finalize(self) -> None:
        count = len(list(self._label_dir.glob("*.txt")))
        logger.info("KITTI labels written: %d frames", count)


class _NuScenesWriter:
    """Writes trajectory annotations in NuScenes-compatible JSON."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self._samples: list[dict[str, Any]] = []

    def add_frame(self, frame: FrameData) -> None:
        sample: dict[str, Any] = {
            "token": f"frame_{frame.frame_id:06d}",
            "timestamp": int(frame.timestamp * 1e6),
            "anns": [],
        }
        for bbox in frame.bounding_boxes:
            sample["anns"].append({
                "category_name": bbox.label,
                "instance_id": bbox.instance_id,
                "bbox_2d": [bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max],
            })
        if frame.agent_positions is not None:
            sample["ego_positions"] = frame.agent_positions.tolist()
        self._samples.append(sample)

    def finalize(self) -> None:
        out_path = self.output_dir / "samples.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"samples": self._samples}, f, indent=2)
        logger.info("NuScenes samples written: %d entries", len(self._samples))


_WRITERS = {
    "coco": _COCOWriter,
    "kitti": _KITTIWriter,
    "nuscenes": _NuScenesWriter,
}


class DataExporter:
    """Orchestrates data capture and export from a simulated scene.

    Parameters
    ----------
    scene : SceneGraph
        The scene to export data from.
    format : str
        Export format (``coco``, ``kitti``, ``nuscenes``, or ``raw``).
    config : ExportConfig, optional
        Full export configuration.
    """

    def __init__(
        self,
        scene: SceneGraph,
        format: str = "coco",
        config: ExportConfig | None = None,
    ) -> None:
        self.scene = scene
        self.config = config or ExportConfig(format=format)
        self._frames: list[FrameData] = []

    def capture_frame(
        self,
        frame_id: int,
        timestamp: float,
        agent_positions: np.ndarray | None = None,
    ) -> FrameData:
        """Capture a single frame of data from the current scene state.

        In a full render pipeline this would invoke the GPU renderer.
        Here we generate synthetic bounding boxes from scene node positions.
        """
        w, h = self.config.resolution
        bboxes: list[BoundingBox2D] = []

        camera_nodes = self.scene.find_by_type(NodeType.CAMERA)
        cam_pos = camera_nodes[0].transform.position if camera_nodes else np.zeros(3)

        for idx, node in enumerate(self.scene.find_by_type(NodeType.PROP)):
            projected = self._project_to_screen(node.transform.position, cam_pos, w, h)
            if projected is None:
                continue
            px, py = projected
            half_w = max(10, int(node.transform.scale[0] * 8))
            half_h = max(10, int(node.transform.scale[2] * 12))

            bbox = BoundingBox2D(
                x_min=max(0, px - half_w),
                y_min=max(0, py - half_h),
                x_max=min(w, px + half_w),
                y_max=min(h, py + half_h),
                label=str(node.properties.get("prop_class", "unknown")),
                instance_id=idx,
            )
            if bbox.area > 0:
                bboxes.append(bbox)

        frame = FrameData(
            frame_id=frame_id,
            timestamp=timestamp,
            bounding_boxes=bboxes,
            agent_positions=agent_positions,
        )
        self._frames.append(frame)
        return frame

    def _project_to_screen(
        self,
        world_pos: np.ndarray,
        cam_pos: np.ndarray,
        width: int,
        height: int,
    ) -> tuple[int, int] | None:
        """Simplified perspective projection (pinhole camera model)."""
        rel = world_pos - cam_pos
        if rel[1] <= 0.1:
            return None  # behind camera

        focal = width / (2 * np.tan(np.radians(30)))
        px = int(width / 2 + focal * rel[0] / rel[1])
        py = int(height / 2 - focal * rel[2] / rel[1])

        if 0 <= px < width and 0 <= py < height:
            return (px, py)
        return None

    def write(self, output_path: str | Path) -> None:
        """Write all captured frames to disk in the configured format."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        fmt = self.config.format
        if fmt == "raw":
            self._write_raw(output_dir)
            return

        writer_cls = _WRITERS.get(fmt)
        if writer_cls is None:
            raise ValueError(f"Unsupported export format: {fmt}")

        if fmt == "coco":
            writer = writer_cls(output_dir, self.config.resolution)
        else:
            writer = writer_cls(output_dir)

        for frame in self._frames:
            writer.add_frame(frame)
        writer.finalize()

        logger.info("Export complete: %d frames -> %s (%s)", len(self._frames), output_dir, fmt)

    def _write_raw(self, output_dir: Path) -> None:
        """Write raw numpy arrays and metadata JSON."""
        for frame in self._frames:
            prefix = output_dir / f"frame_{frame.frame_id:06d}"
            meta = {
                "frame_id": frame.frame_id,
                "timestamp": frame.timestamp,
                "num_boxes": len(frame.bounding_boxes),
            }
            with (prefix.parent / f"{prefix.name}_meta.json").open("w") as f:
                json.dump(meta, f, indent=2)
            if frame.agent_positions is not None:
                np.save(f"{prefix}_agents.npy", frame.agent_positions)

        logger.info("Raw export: %d frames -> %s", len(self._frames), output_dir)

    @property
    def frame_count(self) -> int:
        return len(self._frames)
