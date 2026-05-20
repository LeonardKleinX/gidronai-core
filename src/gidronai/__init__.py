"""GidronAI -- Synthetic Reality Engine.

Generate physically accurate simulated environments for AI training.
"""

from __future__ import annotations

__version__ = "0.4.1"

from gidronai.config import SceneConfig, load_config
from gidronai.engine import SceneEngine, SceneGraph
from gidronai.physics import PhysicsWorld, RigidBody
from gidronai.agents import Agent, AgentPool
from gidronai.export import DataExporter

__all__ = [
    "SceneConfig",
    "SceneEngine",
    "SceneGraph",
    "PhysicsWorld",
    "RigidBody",
    "Agent",
    "AgentPool",
    "DataExporter",
    "load_config",
]
