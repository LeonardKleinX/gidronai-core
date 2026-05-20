"""YAML configuration loader and validation for GidronAI scenes."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import yaml

logger = logging.getLogger(__name__)

_VALID_SCENE_TYPES = frozenset({
    "urban_intersection",
    "highway_segment",
    "parking_lot",
    "indoor_warehouse",
    "residential_street",
    "rural_road",
    "custom",
})

_VALID_EXPORT_FORMATS = frozenset({"coco", "kitti", "nuscenes", "raw"})


@dataclass(frozen=True)
class PhysicsConfig:
    """Physics simulation parameters."""

    gravity: float = -9.81
    dt: float = 1 / 240
    solver_iterations: int = 10
    restitution: float = 0.3
    static_friction: float = 0.5
    dynamic_friction: float = 0.35
    contact_offset: float = 0.002
    enable_soft_body: bool = False

    def effective_substeps(self, target_fps: int = 60) -> int:
        """Return the number of physics substeps per render frame."""
        return max(1, int(1.0 / (self.dt * target_fps)))


@dataclass(frozen=True)
class AgentConfig:
    """Agent population parameters."""

    count: int = 10
    behavior: Literal["social_force", "orca", "boid", "waypoint"] = "social_force"
    spawn_region: str = "sidewalks"
    max_speed: float = 1.8
    neighbor_radius: float = 5.0
    goal_tolerance: float = 0.5
    seed: int | None = None


@dataclass(frozen=True)
class ExportConfig:
    """Data export parameters."""

    format: Literal["coco", "kitti", "nuscenes", "raw"] = "coco"
    modalities: tuple[str, ...] = ("rgb", "depth", "segmentation")
    resolution: tuple[int, int] = (1920, 1080)
    frame_skip: int = 1
    compress: bool = True


@dataclass(frozen=True)
class SimulationConfig:
    """Top-level simulation timing."""

    total_steps: int = 1000
    warmup_steps: int = 50
    target_fps: int = 60


@dataclass(frozen=True)
class SceneConfig:
    """Complete scene configuration."""

    scene_type: str = "urban_intersection"
    size: tuple[float, float, float] = (200.0, 200.0, 50.0)
    weather: str = "clear"
    time_of_day: str = "12:00"
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty if valid)."""
        errors: list[str] = []
        if self.scene_type not in _VALID_SCENE_TYPES:
            errors.append(
                f"Unknown scene_type '{self.scene_type}'. "
                f"Valid: {sorted(_VALID_SCENE_TYPES)}"
            )
        if any(d <= 0 for d in self.size):
            errors.append("All scene dimensions must be positive.")
        if self.physics.dt <= 0:
            errors.append("Physics dt must be positive.")
        if self.agents.count < 0:
            errors.append("Agent count cannot be negative.")
        if self.export.format not in _VALID_EXPORT_FORMATS:
            errors.append(f"Unknown export format '{self.export.format}'.")
        return errors


def _parse_tuple(raw: Any, length: int, dtype: type = float) -> tuple[Any, ...]:
    """Convert a list from YAML into a fixed-length tuple."""
    if isinstance(raw, (list, tuple)):
        if len(raw) != length:
            raise ValueError(f"Expected {length} elements, got {len(raw)}")
        return tuple(dtype(v) for v in raw)
    raise TypeError(f"Expected a list, got {type(raw).__name__}")


def load_config(path: str | Path) -> SceneConfig:
    """Load and validate a YAML scene configuration file.

    Parameters
    ----------
    path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    SceneConfig
        Validated configuration object.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If the configuration contains validation errors.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    logger.debug("Loaded raw config from %s with %d top-level keys", path, len(raw))

    scene_raw = raw.get("scene", {})
    physics_raw = raw.get("physics", {})
    agents_raw = raw.get("agents", {})
    export_raw = raw.get("export", {})
    sim_raw = raw.get("simulation", {})

    physics = PhysicsConfig(
        gravity=physics_raw.get("gravity", -9.81),
        dt=physics_raw.get("dt", 1 / 240),
        solver_iterations=physics_raw.get("solver_iterations", 10),
        restitution=physics_raw.get("restitution", 0.3),
        static_friction=physics_raw.get("static_friction", 0.5),
        dynamic_friction=physics_raw.get("dynamic_friction", 0.35),
        enable_soft_body=physics_raw.get("enable_soft_body", False),
    )

    modalities = export_raw.get("modalities", ["rgb", "depth", "segmentation"])
    resolution = export_raw.get("resolution", [1920, 1080])

    export = ExportConfig(
        format=export_raw.get("format", "coco"),
        modalities=tuple(modalities),
        resolution=_parse_tuple(resolution, 2, int),
        frame_skip=export_raw.get("frame_skip", 1),
        compress=export_raw.get("compress", True),
    )

    agents = AgentConfig(
        count=agents_raw.get("count", 10),
        behavior=agents_raw.get("behavior", "social_force"),
        spawn_region=agents_raw.get("spawn_region", "sidewalks"),
        max_speed=agents_raw.get("max_speed", 1.8),
        seed=agents_raw.get("seed"),
    )

    simulation = SimulationConfig(
        total_steps=sim_raw.get("total_steps", 1000),
        warmup_steps=sim_raw.get("warmup_steps", 50),
        target_fps=sim_raw.get("target_fps", 60),
    )

    size = scene_raw.get("size", [200, 200, 50])

    cfg = SceneConfig(
        scene_type=scene_raw.get("type", "urban_intersection"),
        size=_parse_tuple(size, 3, float),
        weather=scene_raw.get("weather", "clear"),
        time_of_day=scene_raw.get("time_of_day", "12:00"),
        physics=physics,
        agents=agents,
        export=export,
        simulation=simulation,
        metadata=raw.get("metadata", {}),
    )

    errors = cfg.validate()
    if errors:
        raise ValueError(f"Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    logger.info("Scene config loaded: type=%s, agents=%d", cfg.scene_type, cfg.agents.count)
    return cfg
