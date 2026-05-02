"""
agent.py
===========

This module defines the ``Agent`` class used by the longevity society simulation.
Each agent has a unique identifier, a position on the 2D grid, an age and a set
of orientations that influence which place they prefer to visit.  The agent's
decision logic is simple and designed to run efficiently on consumer hardware.

By default, agents use heuristics based on orientation scores.  Optionally,
when ``llm.enabled`` is true in the YAML configuration, one agent per step may
query a local Ollama-compatible server for the next target place; malformed JSON
responses fall back to heuristics and consecutive failures disable LLM via
``Simulation.register_llm_failure``.
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Tuple

if TYPE_CHECKING:
    from simulation import Simulation

logger = logging.getLogger(__name__)


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


def extract_json_object(text: str) -> Optional[dict]:
    """Parse JSON from model output; tolerate prose around or broken fences."""
    raw = text.strip()
    for candidate in (raw, raw.replace("```json", "").replace("```", "")):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        snippet = raw[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


class Agent:
    """
    An agent that moves on a 2D grid and chooses destinations based on
    orientation scores, optionally assisted by a local LLM.
    """

    def __init__(self, agent_id: int, config: dict, places: Dict[str, Place], rng: random.Random):
        self.id: int = agent_id
        self.config = config
        self.places = places
        self.rng = rng

        age_min = config.get("agent_profiles", {}).get("age_min", 20)
        age_max = config.get("agent_profiles", {}).get("age_max", 180)
        self.age: int = rng.randint(age_min, age_max)

        self.family_orientation = rng.uniform(0.0, 1.0)
        self.career_orientation = rng.uniform(0.0, 1.0)
        self.learning_orientation = rng.uniform(0.0, 1.0)
        self.long_term_orientation = rng.uniform(0.0, 1.0)
        self.fertility_desire = rng.uniform(0.0, 1.0)
        self.risk_tolerance = rng.uniform(0.0, 1.0)

        half_space = config["simulation"]["half_space_size"]
        self.x = rng.randint(-half_space, half_space)
        self.y = rng.randint(-half_space, half_space)

        self.current_place: Optional[str] = None
        self.dwell_remaining: int = 0

        self.visit_counts: Dict[str, int] = {name: 0 for name in places.keys()}

    def _distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        x1, y1 = pos1
        x2, y2 = pos2
        return math.hypot(x1 - x2, y1 - y2)

    def step(self, simulation: "Simulation", step: int) -> None:
        """Advance the agent by one time step, updating its position and state."""
        if self.current_place:
            if self.dwell_remaining > 0:
                self.dwell_remaining -= 1
                return
            self.current_place = None
            self._random_move(simulation)
            return

        for name, place in self.places.items():
            if place.contains(self.x, self.y):
                self.current_place = name
                self.visit_counts[name] += 1
                simulation.record_visit(step, self.id, name)
                self.dwell_remaining = self.rng.randint(3, 8)
                return

        if self._try_llm_move(simulation, step):
            return

        self._move_towards_preferred_place(simulation)

    def _ollama_target_place(self) -> Optional[str]:
        """Ask Ollama for a place name; return None on any failure."""
        llm = self.config.get("llm", {})
        base = llm.get("base_url", "http://localhost:11434").rstrip("/")
        model = llm.get("model", "llama3")
        timeout = float(llm.get("timeout", 8.0))
        place_names = ", ".join(sorted(self.places.keys()))
        system = (
            "You help a grid-based simulation. Reply with a single JSON object only, "
            'no other text, shape: {"target_place": "<one of the place names>"}.'
        )
        user = (
            f"Agent id {self.id} at ({self.x},{self.y}), age {self.age}. "
            f"Orientations: family={self.family_orientation:.2f}, career={self.career_orientation:.2f}, "
            f"learning={self.learning_orientation:.2f}, long_term={self.long_term_orientation:.2f}, "
            f"fertility={self.fertility_desire:.2f}, risk={self.risk_tolerance:.2f}.\n"
            f"Valid place names: {place_names}."
        )
        body = json.dumps(
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "format": "json",
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{base}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            logger.debug("Ollama request failed: %s", e)
            return None
        content = (payload.get("message") or {}).get("content")
        if not content or not isinstance(content, str):
            return None
        data = extract_json_object(content)
        if not data:
            return None
        name = data.get("target_place")
        if not isinstance(name, str):
            return None
        name = name.strip()
        if name in self.places:
            return name
        for k in self.places:
            if k.lower() == name.lower():
                return k
        return None

    def _try_llm_move(self, simulation: Simulation, step: int) -> bool:
        if not simulation.llm_effective:
            return False
        n = len(simulation.agents) or 1
        if step % n != self.id:
            return False
        target = self._ollama_target_place()
        if target is None:
            simulation.register_llm_failure()
            return False
        self._move_toward_place(simulation, self.places[target])
        return True

    def _random_move(self, simulation: Simulation) -> None:
        moves = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        dx, dy = self.rng.choice(moves)
        new_x = self.x + dx
        new_y = self.y + dy
        half_space = simulation.config["simulation"]["half_space_size"]
        self.x = max(-half_space, min(half_space, new_x))
        self.y = max(-half_space, min(half_space, new_y))

    def _move_toward_place(self, simulation: Simulation, place: Place) -> None:
        bx, by = place.center
        dx = bx - self.x
        dy = by - self.y
        if dx == 0 and dy == 0:
            return
        if abs(dx) > abs(dy):
            step_x = 1 if dx > 0 else -1
            step_y = 0
        else:
            step_x = 0
            step_y = 1 if dy > 0 else -1
        self.x += step_x
        self.y += step_y
        half_space = self.config["simulation"]["half_space_size"]
        self.x = max(-half_space, min(half_space, self.x))
        self.y = max(-half_space, min(half_space, self.y))

    def _move_towards_preferred_place(self, simulation: Simulation) -> None:
        best_score = -1.0
        best_place: Optional[Place] = None

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
            elif name in ("rejuvenation_clinic", "governance_center", "investment_hub"):
                orient_score = self.long_term_orientation

            dist = self._distance((self.x, self.y), place.center)
            if dist == 0:
                dist = 0.5
            score = orient_score / dist
            if score > best_score:
                best_score = score
                best_place = place

        if best_place is None or best_score < 0.05:
            self._random_move(simulation)
            return

        self._move_toward_place(simulation, best_place)
