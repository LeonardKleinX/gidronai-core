"""Rigid-body physics simulation with configurable integrator."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.spatial import KDTree

from gidronai.config import PhysicsConfig
from gidronai.engine import SceneGraph, SceneNode, NodeType

logger = logging.getLogger(__name__)


@dataclass
class AABB:
    """Axis-aligned bounding box for broad-phase collision detection."""

    min_pt: np.ndarray = field(default_factory=lambda: np.zeros(3))
    max_pt: np.ndarray = field(default_factory=lambda: np.ones(3))

    @property
    def center(self) -> np.ndarray:
        return (self.min_pt + self.max_pt) / 2

    @property
    def half_extents(self) -> np.ndarray:
        return (self.max_pt - self.min_pt) / 2

    def overlaps(self, other: AABB) -> bool:
        return bool(np.all(self.min_pt <= other.max_pt) and np.all(self.max_pt >= other.min_pt))

    def expand(self, margin: float) -> AABB:
        return AABB(
            min_pt=self.min_pt - margin,
            max_pt=self.max_pt + margin,
        )


@dataclass
class RigidBody:
    """A rigid body with mass, velocity, and collision geometry."""

    node: SceneNode
    mass: float = 1.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    force_accumulator: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    is_static: bool = False
    restitution: float = 0.3

    @property
    def position(self) -> np.ndarray:
        return self.node.transform.position

    @position.setter
    def position(self, value: np.ndarray) -> None:
        self.node.transform.position = value

    @property
    def inv_mass(self) -> float:
        return 0.0 if self.is_static else 1.0 / self.mass

    def apply_force(self, force: np.ndarray) -> None:
        if not self.is_static:
            self.force_accumulator += force

    def bounding_box(self) -> AABB:
        half = self.node.transform.scale / 2
        return AABB(
            min_pt=self.position - half,
            max_pt=self.position + half,
        )


@dataclass
class ContactPoint:
    """Result of a narrow-phase collision test."""

    body_a: RigidBody
    body_b: RigidBody
    normal: np.ndarray
    penetration: float
    point: np.ndarray


class PhysicsWorld:
    """Manages the physics simulation loop.

    Parameters
    ----------
    scene : SceneGraph
        The scene to simulate.
    gravity : float
        Gravitational acceleration along the Z axis.
    dt : float
        Fixed timestep in seconds.
    solver_iterations : int
        Number of constraint solver iterations per step.
    """

    def __init__(
        self,
        scene: SceneGraph,
        gravity: float = -9.81,
        dt: float = 1 / 240,
        solver_iterations: int = 10,
        config: PhysicsConfig | None = None,
    ) -> None:
        self._config = config or PhysicsConfig(gravity=gravity, dt=dt, solver_iterations=solver_iterations)
        self.scene = scene
        self.bodies: list[RigidBody] = []
        self.time: float = 0.0
        self._step_count: int = 0
        self._gravity_vec = np.array([0.0, 0.0, self._config.gravity])

        self._register_scene_bodies()

    def _register_scene_bodies(self) -> None:
        """Create rigid bodies for all structure and prop nodes."""
        for node in self.scene.find_by_type(NodeType.STRUCTURE):
            body = RigidBody(node=node, is_static=True, mass=1e6)
            self.bodies.append(body)

        for node in self.scene.find_by_type(NodeType.PROP):
            mass = 5.0 if node.properties.get("prop_class") == "bench" else 1.0
            body = RigidBody(node=node, is_static=True, mass=mass)
            self.bodies.append(body)

        logger.info("Registered %d physics bodies", len(self.bodies))

    def add_body(self, body: RigidBody) -> None:
        self.bodies.append(body)

    def step(self) -> None:
        """Advance the simulation by one fixed timestep."""
        dt = self._config.dt

        # Apply gravity
        for body in self.bodies:
            if not body.is_static:
                body.apply_force(self._gravity_vec * body.mass)

        # Semi-implicit Euler integration
        for body in self.bodies:
            if body.is_static:
                continue
            acceleration = body.force_accumulator * body.inv_mass
            body.velocity += acceleration * dt
            body.position = body.position + body.velocity * dt
            body.force_accumulator[:] = 0.0

        # Broad-phase collision detection
        contacts = self._detect_collisions()

        # Solve contacts
        for _ in range(self._config.solver_iterations):
            for contact in contacts:
                self._resolve_contact(contact)

        # Ground plane constraint (z >= 0)
        for body in self.bodies:
            if not body.is_static and body.position[2] < 0:
                body.position[2] = 0.0
                body.velocity[2] = -body.velocity[2] * body.restitution

        self.time += dt
        self._step_count += 1

    def _detect_collisions(self) -> list[ContactPoint]:
        """Broad + narrow phase collision detection."""
        dynamic = [b for b in self.bodies if not b.is_static]
        if not dynamic:
            return []

        contacts: list[ContactPoint] = []
        positions = np.array([b.position for b in self.bodies])

        if len(positions) < 2:
            return contacts

        tree = KDTree(positions)
        checked: set[tuple[int, int]] = set()

        for i, body_a in enumerate(self.bodies):
            if body_a.is_static:
                continue
            radius = float(np.max(body_a.node.transform.scale))
            neighbors = tree.query_ball_point(body_a.position, radius * 1.5)

            for j in neighbors:
                if i == j:
                    continue
                pair = (min(i, j), max(i, j))
                if pair in checked:
                    continue
                checked.add(pair)

                body_b = self.bodies[j]
                contact = self._narrow_phase(body_a, body_b)
                if contact is not None:
                    contacts.append(contact)

        return contacts

    def _narrow_phase(self, a: RigidBody, b: RigidBody) -> ContactPoint | None:
        """AABB overlap test and penetration calculation."""
        box_a = a.bounding_box()
        box_b = b.bounding_box()

        if not box_a.overlaps(box_b):
            return None

        delta = b.position - a.position
        dist = float(np.linalg.norm(delta))
        if dist < 1e-8:
            normal = np.array([0.0, 0.0, 1.0])
            dist = 1e-8
        else:
            normal = delta / dist

        overlap = float(np.sum(box_a.half_extents + box_b.half_extents)) / 3 - dist
        if overlap <= 0:
            return None

        return ContactPoint(
            body_a=a,
            body_b=b,
            normal=normal,
            penetration=overlap,
            point=(a.position + b.position) / 2,
        )

    def _resolve_contact(self, contact: ContactPoint) -> None:
        """Impulse-based contact resolution."""
        a, b = contact.body_a, contact.body_b
        relative_vel = a.velocity - b.velocity
        vel_along_normal = float(np.dot(relative_vel, contact.normal))

        if vel_along_normal > 0:
            return  # separating

        restitution = min(a.restitution, b.restitution)
        inv_mass_sum = a.inv_mass + b.inv_mass
        if inv_mass_sum == 0:
            return

        impulse_mag = -(1 + restitution) * vel_along_normal / inv_mass_sum
        impulse = impulse_mag * contact.normal

        if not a.is_static:
            a.velocity += impulse * a.inv_mass
        if not b.is_static:
            b.velocity -= impulse * b.inv_mass

        # Positional correction (Baumgarte stabilization)
        correction_pct = 0.2
        slop = 0.01
        correction_mag = max(contact.penetration - slop, 0.0) / inv_mass_sum * correction_pct
        correction = correction_mag * contact.normal

        if not a.is_static:
            a.position = a.position - correction * a.inv_mass
        if not b.is_static:
            b.position = b.position + correction * b.inv_mass

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def kinetic_energy(self) -> float:
        """Total kinetic energy of all dynamic bodies."""
        total = 0.0
        for body in self.bodies:
            if not body.is_static:
                speed_sq = float(np.dot(body.velocity, body.velocity))
                total += 0.5 * body.mass * speed_sq
        return total
