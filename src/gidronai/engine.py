"""Scene graph construction and procedural synthesis engine."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterator, Sequence

import numpy as np

from gidronai.config import SceneConfig

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Classification of scene graph nodes."""

    ROOT = auto()
    TERRAIN = auto()
    STRUCTURE = auto()
    PROP = auto()
    LIGHT = auto()
    CAMERA = auto()
    SPAWN_REGION = auto()


@dataclass
class Transform:
    """3D transform with position, rotation (quaternion), and scale."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    rotation: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    scale: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float64))

    def to_matrix(self) -> np.ndarray:
        """Convert to a 4x4 homogeneous transformation matrix."""
        w, x, y, z = self.rotation
        mat = np.eye(4, dtype=np.float64)
        mat[0, 0] = 1 - 2 * (y * y + z * z)
        mat[0, 1] = 2 * (x * y - z * w)
        mat[0, 2] = 2 * (x * z + y * w)
        mat[1, 0] = 2 * (x * y + z * w)
        mat[1, 1] = 1 - 2 * (x * x + z * z)
        mat[1, 2] = 2 * (y * z - x * w)
        mat[2, 0] = 2 * (x * z - y * w)
        mat[2, 1] = 2 * (y * z + x * w)
        mat[2, 2] = 1 - 2 * (x * x + y * y)
        mat[:3, :3] *= self.scale[:, np.newaxis]
        mat[:3, 3] = self.position
        return mat


@dataclass
class SceneNode:
    """A single node in the scene graph."""

    uid: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    node_type: NodeType = NodeType.PROP
    transform: Transform = field(default_factory=Transform)
    children: list[SceneNode] = field(default_factory=list)
    properties: dict[str, object] = field(default_factory=dict)

    def walk(self) -> Iterator[SceneNode]:
        """Depth-first traversal of this subtree."""
        yield self
        for child in self.children:
            yield from child.walk()

    @property
    def descendant_count(self) -> int:
        return sum(1 for _ in self.walk()) - 1


class SceneGraph:
    """Hierarchical scene representation built during synthesis."""

    def __init__(self, root: SceneNode | None = None) -> None:
        self.root = root or SceneNode(name="world", node_type=NodeType.ROOT)
        self._index: dict[str, SceneNode] = {}
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._index = {node.uid: node for node in self.root.walk()}

    def add(self, node: SceneNode, parent_uid: str | None = None) -> None:
        """Attach a node to the graph under the specified parent."""
        parent = self._index.get(parent_uid or self.root.uid, self.root)
        parent.children.append(node)
        for n in node.walk():
            self._index[n.uid] = n

    def find_by_type(self, node_type: NodeType) -> list[SceneNode]:
        """Return all nodes of the given type."""
        return [n for n in self._index.values() if n.node_type == node_type]

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, uid: str) -> bool:
        return uid in self._index

    def get(self, uid: str) -> SceneNode | None:
        return self._index.get(uid)


class SceneEngine:
    """Procedural scene synthesis engine.

    Given a ``SceneConfig``, the engine builds a ``SceneGraph`` by sequentially
    running terrain generation, structure placement, prop scattering, lighting
    setup, and camera rig creation.
    """

    def __init__(self, config: SceneConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.metadata.get("seed", 42))

    def synthesize(self) -> SceneGraph:
        """Run the full synthesis pipeline and return the scene graph."""
        logger.info("Synthesizing scene: type=%s", self.config.scene_type)

        graph = SceneGraph()
        self._generate_terrain(graph)
        self._place_structures(graph)
        self._scatter_props(graph)
        self._setup_lighting(graph)
        self._create_cameras(graph)
        self._mark_spawn_regions(graph)

        logger.info("Scene synthesis complete: %d nodes", len(graph))
        return graph

    def _generate_terrain(self, graph: SceneGraph) -> None:
        sx, sy, _ = self.config.size
        heightmap = self._rng.normal(0, 0.15, size=(64, 64)).astype(np.float32)

        terrain = SceneNode(
            name="terrain",
            node_type=NodeType.TERRAIN,
            properties={
                "heightmap_shape": heightmap.shape,
                "bounds": (sx, sy),
                "material": "asphalt" if "urban" in self.config.scene_type else "grass",
            },
        )
        terrain.transform.scale = np.array([sx, sy, 1.0])
        graph.add(terrain)

    def _place_structures(self, graph: SceneGraph) -> None:
        sx, sy, sz = self.config.size
        num_structures = max(1, int(sx * sy / 2000))
        for i in range(num_structures):
            pos = self._rng.uniform([-sx / 2, -sy / 2, 0], [sx / 2, sy / 2, 0])
            height = self._rng.uniform(3.0, min(sz, 30.0))
            width = self._rng.uniform(5.0, 20.0)

            node = SceneNode(
                name=f"structure_{i:03d}",
                node_type=NodeType.STRUCTURE,
                transform=Transform(position=pos, scale=np.array([width, width, height])),
                properties={"floors": max(1, int(height / 3.0))},
            )
            graph.add(node)

    def _scatter_props(self, graph: SceneGraph) -> None:
        sx, sy, _ = self.config.size
        prop_density = 0.005  # props per square meter
        num_props = int(sx * sy * prop_density)
        prop_types = ["tree", "bench", "lamp_post", "sign", "bollard", "trash_can"]

        for i in range(num_props):
            pos = self._rng.uniform([-sx / 2, -sy / 2, 0], [sx / 2, sy / 2, 0])
            prop_type = self._rng.choice(prop_types)
            angle = self._rng.uniform(0, 2 * np.pi)
            quat = np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])

            node = SceneNode(
                name=f"prop_{prop_type}_{i:04d}",
                node_type=NodeType.PROP,
                transform=Transform(position=pos, rotation=quat),
                properties={"prop_class": prop_type},
            )
            graph.add(node)

    def _setup_lighting(self, graph: SceneGraph) -> None:
        hour, minute = (int(x) for x in self.config.time_of_day.split(":"))
        sun_angle = (hour + minute / 60 - 6) / 12 * np.pi  # 6am=0, 6pm=pi
        sun_elevation = max(0.0, np.sin(sun_angle))

        intensity = 1.0 if self.config.weather == "clear" else 0.4
        sun = SceneNode(
            name="sun_light",
            node_type=NodeType.LIGHT,
            transform=Transform(
                position=np.array([0.0, 0.0, 100.0]),
            ),
            properties={
                "light_type": "directional",
                "intensity": intensity * sun_elevation,
                "color_temperature": 5500 + int(1500 * (1 - sun_elevation)),
                "cast_shadows": True,
            },
        )
        graph.add(sun)

        ambient = SceneNode(
            name="ambient_light",
            node_type=NodeType.LIGHT,
            properties={
                "light_type": "ambient",
                "intensity": 0.15 + 0.1 * (1 - sun_elevation),
            },
        )
        graph.add(ambient)

    def _create_cameras(self, graph: SceneGraph) -> None:
        w, h = self.config.export.resolution
        fov_y = 60.0
        aspect = w / h

        cam = SceneNode(
            name="main_camera",
            node_type=NodeType.CAMERA,
            transform=Transform(position=np.array([0.0, -30.0, 15.0])),
            properties={
                "fov_y": fov_y,
                "aspect_ratio": aspect,
                "near_clip": 0.1,
                "far_clip": 1000.0,
                "resolution": (w, h),
            },
        )
        graph.add(cam)

    def _mark_spawn_regions(self, graph: SceneGraph) -> None:
        sx, sy, _ = self.config.size
        region = self.config.agents.spawn_region

        if region == "sidewalks":
            bounds = [(-sx / 2, -sy / 2 + 2, sx / 2, -sy / 2 + 5)]
        elif region == "full":
            bounds = [(-sx / 2, -sy / 2, sx / 2, sy / 2)]
        else:
            bounds = [(-sx / 4, -sy / 4, sx / 4, sy / 4)]

        for idx, (x0, y0, x1, y1) in enumerate(bounds):
            node = SceneNode(
                name=f"spawn_region_{idx}",
                node_type=NodeType.SPAWN_REGION,
                properties={"bounds_2d": (x0, y0, x1, y1)},
            )
            graph.add(node)
