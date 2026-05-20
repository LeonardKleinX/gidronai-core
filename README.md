<p align="center">
  <img src="https://gidronai.me/assets/logo-dark.svg" alt="GidronAI" width="320" />
</p>

<h3 align="center">Synthetic Reality Engine</h3>

<p align="center">
  <a href="https://pypi.org/project/gidronai/"><img src="https://img.shields.io/pypi/v/gidronai?color=blue" alt="PyPI" /></a>
  <a href="https://github.com/LeonardKleinX/gidronai-core/actions"><img src="https://img.shields.io/github/actions/workflow/status/LeonardKleinX/gidronai-core/ci.yml?branch=main" alt="CI" /></a>
  <a href="https://github.com/LeonardKleinX/gidronai-core/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License" /></a>
  <a href="https://docs.gidronai.me"><img src="https://img.shields.io/badge/docs-latest-brightgreen" alt="Docs" /></a>
  <a href="https://pypi.org/project/gidronai/"><img src="https://img.shields.io/pypi/pyversions/gidronai" alt="Python" /></a>
</p>

---

**GidronAI Core** is an open-source Python engine for generating physically accurate simulated environments. It enables researchers and engineers to synthesize photorealistic 3D scenes with configurable physics, populate them with autonomous agents, and export structured training data for downstream AI/ML pipelines.

## Features

- **Scene Synthesis** -- Procedural generation of indoor/outdoor environments with parametric control over geometry, materials, lighting, and weather conditions.
- **Physics Simulation** -- Rigid-body and soft-body dynamics powered by a custom integrator with configurable gravity, friction, and collision models.
- **Multi-Agent Behavior** -- Spawn and orchestrate autonomous agents with goal-driven navigation, obstacle avoidance, and social force dynamics.
- **Data Export Pipeline** -- First-class support for exporting RGB, depth, segmentation masks, bounding boxes, and trajectory logs in COCO, KITTI, and NuScenes formats.
- **YAML-Driven Configuration** -- Every parameter is configurable via a single YAML file. No code changes needed to run new experiments.

## Installation

```bash
pip install gidronai
```

For development:

```bash
git clone https://github.com/LeonardKleinX/gidronai-core.git
cd gidronai-core
pip install -e ".[dev]"
```

**Requirements:** Python 3.10+, NumPy >= 1.24, PyYAML >= 6.0

## Quickstart

```python
from gidronai import SceneEngine, PhysicsWorld, AgentPool
from gidronai.config import load_config

# Load scene configuration
cfg = load_config("scene.yaml")

# Build the environment
engine = SceneEngine(cfg)
scene = engine.synthesize()

# Attach physics
world = PhysicsWorld(scene, gravity=-9.81, dt=1/240)

# Spawn agents
pool = AgentPool(scene, num_agents=12, behavior="social_force")
pool.assign_goals_from_config(cfg)

# Run simulation
for step in range(cfg.simulation.total_steps):
    world.step()
    pool.update(world.time)

# Export training data
from gidronai.export import DataExporter
exporter = DataExporter(scene, format="coco")
exporter.write("./output/training_data")
```

## Architecture

```
gidronai/
  engine.py      -- Scene graph construction and procedural synthesis
  physics.py     -- Rigid/soft-body physics world and collision detection
  agents.py      -- Agent spawning, navigation, and behavior trees
  export.py      -- Multi-format data export (COCO, KITTI, NuScenes)
  config.py      -- YAML config loader and validation
```

The engine follows a **compose-then-simulate** pattern: build a static scene graph, attach physics constraints, inject agents, run the time loop, and export. Each stage is independently configurable and can be used in isolation.

## Configuration

All experiments are driven by YAML configs. See [`gidronai-examples`](https://github.com/LeonardKleinX/gidronai-examples) for templates.

```yaml
scene:
  type: urban_intersection
  size: [200, 200, 50]
  weather: overcast
  time_of_day: 14:30

physics:
  gravity: -9.81
  dt: 0.004167
  solver_iterations: 10

agents:
  count: 24
  behavior: social_force
  spawn_region: sidewalks

export:
  format: coco
  modalities: [rgb, depth, segmentation]
  resolution: [1920, 1080]
```

## Documentation

Full API reference and tutorials are available at [docs.gidronai.me](https://docs.gidronai.me).

## Contributing

We welcome contributions. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License. See [LICENSE](LICENSE) for details.

---

Built by [GidronAI](https://gidronai.me) -- Toronto, Canada.
