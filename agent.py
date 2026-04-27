"""
agent.py
===========

This module defines the ``Agent`` class used by the longevity society simulation.
Each agent has a unique identifier, a position on the 2D grid, an age and a set
of orientations that influence which place they prefer to visit.  The agent's
decision logic is simple and designed to run efficiently on consumer hardware.

Agents do not call large language models.  Instead, they use heuristics based
on their internal orientation scores to choose a target destination.  When an
agent is inside a place, they remain there for a random dwell time before
exiting.  Outside of places, they move towards their preferred destination or
wander randomly.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Place:
    """Represents a square region that agents can occupy."""

    name: str
    center: Tuple[int, int]
    half_size: int

    def contains(self, x: int, y: int) -> bool:
        cx, cy = self.center
        return (
            cx - self.half_size <= x <= cx + self.half_size
            and cy - self.half_size <= y <= cy + self.half_size
        )


class Agent:
    """
    An agent that moves on a 2D grid and chooses destinations based on
    orientation scores.

    :param agent_id: Unique integer identifier for the agent.
    :param config: Parsed YAML configuration dictionary.
    :param places: Mapping from place names to ``Place`` objects.
    :param rng: A random.Random instance for reproducibility.
    """

    def __init__(self, agent_id: int, config: dict, places: Dict[str, Place], rng: random.Random):
        self.id: int = agent_id
        self.config = config
        self.places = places
        self.rng = rng

        # Sample the agent's initial attributes
        age_min = config.get("agent_profiles", {}).get("age_min", 20)
        age_max = config.get("agent_profiles", {}).get("age_max", 180)
        self.age: int = rng.randint(age_min, age_max)

        # Orientation scores between 0.0 and 1.0
        self.family_orientation = rng.uniform(0.0, 1.0)
        self.career_orientation = rng.uniform(0.0, 1.0)
        self.learning_orientation = rng.uniform(0.0, 1.0)
        self.long_term_orientation = rng.uniform(0.0, 1.0)
        self.fertility_desire = rng.uniform(0.0, 1.0)
        self.risk_tolerance = rng.uniform(0.0, 1.0)

        # Current position on the grid
        half_space = config["simulation"]["half_space_size"]
        self.x = rng.randint(-half_space, half_space)
        self.y = rng.randint(-half_space, half_space)

        # State for dwell time if inside a place
        self.current_place: Optional[str] = None
        self.dwell_remaining: int = 0

        # Count visits to each place
        self.visit_counts: Dict[str, int] = {name: 0 for name in places.keys()}

    def _distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        x1, y1 = pos1
        x2, y2 = pos2
        return math.hypot(x1 - x2, y1 - y2)

    def step(self, simulation: "Simulation") -> None:
        """Advance the agent by one time step, updating its position and state."""
        # If currently inside a place, decrement dwell time and leave when done
        if self.current_place:
            if self.dwell_remaining > 0:
                self.dwell_remaining -= 1
                return
            else:
                # Leave the place: simply mark as no longer inside
                self.current_place = None
                # random step outside after leaving
                self._random_move(simulation)
                return

        # Check if agent has entered any place at current position
        for name, place in self.places.items():
            if place.contains(self.x, self.y):
                # Enter the place
                self.current_place = name
                self.visit_counts[name] += 1
                # Choose a random dwell time between 3 and 8 steps
                self.dwell_remaining = self.rng.randint(3, 8)
                return

        # Otherwise, decide where to go
        self._move_towards_preferred_place()

    def _random_move(self, simulation: "Simulation") -> None:
        """Move randomly one unit in any cardinal direction or stay in place."""
        moves = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        dx, dy = self.rng.choice(moves)
        new_x = self.x + dx
        new_y = self.y + dy
        # keep within bounds
        half_space = simulation.config["simulation"]["half_space_size"]
        self.x = max(-half_space, min(half_space, new_x))
        self.y = max(-half_space, min(half_space, new_y))

    def _move_towards_preferred_place(self) -> None:
        """
        Move one step towards the most preferred place based on orientation scores.
        This heuristic computes a simple score for each place based on the agent's
        orientation and the distance to that place.  The agent then moves
        one unit towards the highest scoring place.  If no place is strongly
        preferred, the agent will wander randomly.
        """
        best_score = -1.0
        best_place: Optional[Place] = None

        # Assign weights to orientations per place
        for name, place in self.places.items():
            orient_score = 0.0
            if name == "family_zone":
                orient_score = self.family_orientation + self.fertility_desire
            elif name == "academy":
                orient_score = self.learning_orientation
            elif name == "workplace":
                orient_score = self.career_orientation
            elif name == "startup_lab":
                orient_score = (self.career_orientation + self.risk_tolerance) / 2.0
            elif name == "rejuvenation_clinic":
                orient_score = self.long_term_orientation
            elif name == "governance_center":
                orient_score = self.long_term_orientation
            elif name == "investment_hub":
                orient_score = self.long_term_orientation

            # compute distance to place center
            dist = self._distance((self.x, self.y), place.center)
            # Avoid division by zero and penalise distant targets
            if dist == 0:
                dist = 0.5
            score = orient_score / dist
            if score > best_score:
                best_score = score
                best_place = place

        # Threshold for decision: if the best score is very low, wander randomly
        if best_place is None or best_score < 0.05:
            # wander randomly
            self._random_move(simulation)
            return

        # Move one step towards the best place center
        bx, by = best_place.center
        dx = bx - self.x
        dy = by - self.y
        # normalise to one step
        if abs(dx) > abs(dy):
            step_x = 1 if dx > 0 else -1
            step_y = 0
        else:
            step_x = 0
            step_y = 1 if dy > 0 else -1
        self.x += step_x
        self.y += step_y

        # ensure we stay within world bounds
        half_space = self.config["simulation"]["half_space_size"]
        self.x = max(-half_space, min(half_space, self.x))
        self.y = max(-half_space, min(half_space, self.y))