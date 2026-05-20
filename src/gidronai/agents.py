"""Multi-agent behavior simulation with pluggable navigation models."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Sequence

import numpy as np

from gidronai.config import AgentConfig, SceneConfig
from gidronai.engine import SceneGraph, SceneNode, NodeType, Transform

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Lifecycle states for a simulated agent."""

    IDLE = auto()
    NAVIGATING = auto()
    ARRIVED = auto()
    BLOCKED = auto()


@dataclass
class Agent:
    """An autonomous agent within the simulated environment.

    Each agent has a position, velocity, goal, and behavioral parameters
    controlled by the selected navigation model.
    """

    agent_id: int
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    goal: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    max_speed: float = 1.8
    preferred_speed: float = 1.4
    radius: float = 0.3
    state: AgentState = AgentState.IDLE
    path: list[np.ndarray] = field(default_factory=list)

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    def distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.goal - self.position))


class SocialForceModel:
    """Social Force Model (Helbing & Molnar, 1995).

    Computes steering forces from three components:
    1. Desired-velocity force (goal attraction)
    2. Agent-agent repulsion (personal space)
    3. Obstacle repulsion (wall avoidance)
    """

    def __init__(
        self,
        tau: float = 0.5,
        agent_repulsion_strength: float = 2.1,
        agent_repulsion_range: float = 0.3,
        obstacle_repulsion_strength: float = 10.0,
        obstacle_repulsion_range: float = 0.2,
    ) -> None:
        self.tau = tau
        self.a_strength = agent_repulsion_strength
        self.a_range = agent_repulsion_range
        self.o_strength = obstacle_repulsion_strength
        self.o_range = obstacle_repulsion_range

    def compute_force(
        self,
        agent: Agent,
        neighbors: Sequence[Agent],
        obstacles: Sequence[np.ndarray],
    ) -> np.ndarray:
        """Compute the total social force acting on this agent."""
        f_desired = self._desired_force(agent)
        f_agents = self._agent_repulsion(agent, neighbors)
        f_obstacles = self._obstacle_repulsion(agent, obstacles)
        return f_desired + f_agents + f_obstacles

    def _desired_force(self, agent: Agent) -> np.ndarray:
        direction = agent.goal - agent.position
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            return np.zeros(3)
        desired_vel = (direction / dist) * agent.preferred_speed
        return (desired_vel - agent.velocity) / self.tau

    def _agent_repulsion(self, agent: Agent, neighbors: Sequence[Agent]) -> np.ndarray:
        force = np.zeros(3, dtype=np.float64)
        for other in neighbors:
            if other.agent_id == agent.agent_id:
                continue
            diff = agent.position - other.position
            dist = float(np.linalg.norm(diff))
            if dist < 1e-6:
                diff = np.array([1e-3, 0, 0])
                dist = 1e-3
            overlap = agent.radius + other.radius - dist
            normal = diff / dist
            magnitude = self.a_strength * np.exp(overlap / self.a_range)
            force += magnitude * normal
        return force

    def _obstacle_repulsion(self, agent: Agent, obstacles: Sequence[np.ndarray]) -> np.ndarray:
        force = np.zeros(3, dtype=np.float64)
        for obs_pos in obstacles:
            diff = agent.position - obs_pos
            dist = float(np.linalg.norm(diff))
            if dist < 1e-6:
                continue
            normal = diff / dist
            magnitude = self.o_strength * np.exp(-dist / self.o_range)
            force += magnitude * normal
        return force


class ORCAModel:
    """Optimal Reciprocal Collision Avoidance (simplified 2D projection)."""

    def __init__(self, time_horizon: float = 3.0, neighbor_dist: float = 5.0) -> None:
        self.time_horizon = time_horizon
        self.neighbor_dist = neighbor_dist

    def compute_force(
        self,
        agent: Agent,
        neighbors: Sequence[Agent],
        obstacles: Sequence[np.ndarray],
    ) -> np.ndarray:
        preferred_vel = np.zeros(3)
        direction = agent.goal - agent.position
        dist = np.linalg.norm(direction)
        if dist > 1e-6:
            preferred_vel = (direction / dist) * agent.preferred_speed

        # Simplified ORCA: adjust preferred velocity away from neighbors
        adjusted = preferred_vel.copy()
        for other in neighbors:
            if other.agent_id == agent.agent_id:
                continue
            rel_pos = other.position - agent.position
            rel_dist = float(np.linalg.norm(rel_pos))
            if rel_dist > self.neighbor_dist or rel_dist < 1e-6:
                continue
            combined_radius = agent.radius + other.radius
            if rel_dist < combined_radius * 2:
                avoidance = -rel_pos / rel_dist
                weight = 1.0 - (rel_dist / (combined_radius * 2))
                adjusted += avoidance * weight * agent.max_speed

        return (adjusted - agent.velocity) * 2.0


_BEHAVIOR_MODELS = {
    "social_force": SocialForceModel,
    "orca": ORCAModel,
}


class AgentPool:
    """Manages a population of agents within a scene.

    Parameters
    ----------
    scene : SceneGraph
        The scene graph containing spawn regions and obstacles.
    num_agents : int
        Number of agents to spawn.
    behavior : str
        Navigation model name (``social_force`` or ``orca``).
    config : AgentConfig, optional
        Full agent configuration.
    """

    def __init__(
        self,
        scene: SceneGraph,
        num_agents: int = 10,
        behavior: str = "social_force",
        config: AgentConfig | None = None,
    ) -> None:
        self.scene = scene
        self.config = config or AgentConfig(count=num_agents, behavior=behavior)
        self._rng = np.random.default_rng(self.config.seed)

        model_cls = _BEHAVIOR_MODELS.get(self.config.behavior)
        if model_cls is None:
            raise ValueError(f"Unknown behavior model: {self.config.behavior}")
        self.model = model_cls()

        self.agents: list[Agent] = []
        self._obstacle_positions: list[np.ndarray] = []
        self._init_obstacles()
        self._spawn_agents()

    def _init_obstacles(self) -> None:
        for node in self.scene.find_by_type(NodeType.STRUCTURE):
            self._obstacle_positions.append(node.transform.position.copy())
        for node in self.scene.find_by_type(NodeType.PROP):
            self._obstacle_positions.append(node.transform.position.copy())

    def _spawn_agents(self) -> None:
        spawn_regions = self.scene.find_by_type(NodeType.SPAWN_REGION)
        if not spawn_regions:
            logger.warning("No spawn regions found; spawning at origin")
            bounds = (-10, -10, 10, 10)
        else:
            bounds = spawn_regions[0].properties.get("bounds_2d", (-10, -10, 10, 10))

        x0, y0, x1, y1 = bounds
        for i in range(self.config.count):
            pos = np.array([
                self._rng.uniform(x0, x1),
                self._rng.uniform(y0, y1),
                0.0,
            ])
            goal = np.array([
                self._rng.uniform(x0, x1),
                self._rng.uniform(y0, y1),
                0.0,
            ])
            agent = Agent(
                agent_id=i,
                position=pos,
                goal=goal,
                max_speed=self.config.max_speed,
                preferred_speed=self.config.max_speed * 0.78,
                state=AgentState.NAVIGATING,
            )
            self.agents.append(agent)

        logger.info("Spawned %d agents with %s model", len(self.agents), self.config.behavior)

    def assign_goals_from_config(self, config: SceneConfig) -> None:
        """Reassign random goals within the scene bounds."""
        sx, sy, _ = config.size
        for agent in self.agents:
            agent.goal = np.array([
                self._rng.uniform(-sx / 2, sx / 2),
                self._rng.uniform(-sy / 2, sy / 2),
                0.0,
            ])
            agent.state = AgentState.NAVIGATING

    def update(self, time: float, dt: float = 1 / 60) -> None:
        """Advance all agents by one tick."""
        for agent in self.agents:
            if agent.state == AgentState.ARRIVED:
                continue

            if agent.distance_to_goal() < self.config.goal_tolerance:
                agent.state = AgentState.ARRIVED
                agent.velocity[:] = 0.0
                continue

            neighbors = [
                a for a in self.agents
                if a.agent_id != agent.agent_id
                and np.linalg.norm(a.position - agent.position) < self.config.neighbor_radius
            ]

            force = self.model.compute_force(agent, neighbors, self._obstacle_positions)
            agent.velocity += force * dt

            speed = np.linalg.norm(agent.velocity)
            if speed > agent.max_speed:
                agent.velocity = (agent.velocity / speed) * agent.max_speed

            agent.position += agent.velocity * dt

    @property
    def active_count(self) -> int:
        return sum(1 for a in self.agents if a.state == AgentState.NAVIGATING)

    def positions_array(self) -> np.ndarray:
        """Return an (N, 3) array of all agent positions."""
        return np.array([a.position for a in self.agents])

    def velocities_array(self) -> np.ndarray:
        """Return an (N, 3) array of all agent velocities."""
        return np.array([a.velocity for a in self.agents])
