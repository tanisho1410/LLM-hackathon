"""
simulation.py
===============

This module implements a simple grid‐based simulation for a longevity
society.  It reads a YAML configuration describing the world bounds,
places (such as academies, workplaces and family zones), the number of
agents and the duration of the run.  Agents move on the grid using
heuristics defined in ``agent.py`` and record visits to places.

The simulation is intentionally lightweight: it does not call large
language models or perform expensive reasoning at each step.  It is
designed to run on a consumer PC and generate reproducible
observations.

Usage::

    from simulation import Simulation
    sim = Simulation('config.yaml', seed=42)
    sim.run()

Results are logged to a file (simulation.log by default) and a summary
is printed at the end of the run.
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml

from agent import Agent, Place


@dataclass
class Event:
    """Represents a time‐triggered event.  Unused in this simple simulation."""

    name: str
    start_step: int
    intensity: float
    radius: int
    center_x: int
    center_y: int

    def is_active(self, step: int) -> bool:
        return step >= self.start_step


class Simulation:
    """
    Controls the execution of the agent simulation.  Loads configuration
    from YAML, initialises agents and places, and advances the system
    step by step.
    """

    def __init__(self, config_path: str, seed: Optional[int] = None) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            self.config: dict = yaml.safe_load(f)

        # Setup logging
        log_file = self.config.get("logging", {}).get("log_file", "simulation.log")
        logging.basicConfig(
            filename=log_file,
            level=getattr(logging, self.config.get("logging", {}).get("level", "INFO")),
            filemode="w",
            format="%(asctime)s %(levelname)s:%(message)s",
        )
        logging.info("Initialising simulation")

        # Random seed for reproducibility
        self.rng = random.Random(seed)

        # Create place objects
        self.places: Dict[str, Place] = {}
        for place_cfg in self.config.get("places", []):
            name = place_cfg["name"]
            center_x = place_cfg.get("center_x", 0)
            center_y = place_cfg.get("center_y", 0)
            half_size = place_cfg.get("half_size", self.config["simulation"]["half_place_size"])
            self.places[name] = Place(
                name=name,
                center=(center_x, center_y),
                half_size=half_size,
            )

        # Create event objects (not used in this simple example)
        self.events: List[Event] = []
        for event_cfg in self.config.get("fires", []):
            self.events.append(
                Event(
                    name=event_cfg["name"],
                    start_step=event_cfg.get("start_step", 0),
                    intensity=event_cfg.get("intensity", 0.0),
                    radius=event_cfg.get("radius", 0),
                    center_x=event_cfg.get("center_x", 0),
                    center_y=event_cfg.get("center_y", 0),
                )
            )

        # Create agents
        num_agents = self.config.get("agents", {}).get("num_agents", 10)
        self.agents: List[Agent] = []
        for i in range(num_agents):
            agent = Agent(agent_id=i, config=self.config, places=self.places, rng=self.rng)
            self.agents.append(agent)

    def run(self) -> None:
        """Execute the simulation."""
        duration = self.config["simulation"]["duration"]
        logging.info(f"Running simulation for {duration} steps with {len(self.agents)} agents")
        start_time = time.time()
        # Statistics: total visits per place across agents
        global_visits: Dict[str, int] = {name: 0 for name in self.places.keys()}
        # Main loop
        for step in range(duration):
            # For each agent, advance one step
            for agent in self.agents:
                agent.step(self)
            # Optionally update event effects here (not implemented)
        # Aggregate visits after simulation
        for agent in self.agents:
            for name, count in agent.visit_counts.items():
                global_visits[name] += count
        end_time = time.time()
        elapsed = end_time - start_time
        logging.info("Simulation completed")
        logging.info(f"Elapsed time: {elapsed:.2f} seconds")
        # Print summary to stdout
        print("Simulation complete.")
        print(f"Duration: {duration} steps, agents: {len(self.agents)}, time: {elapsed:.2f}s")
        print("Total visits per place:")
        for name, total in global_visits.items():
            print(f"  {name}: {total}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run longevity society simulation")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    sim = Simulation(args.config, seed=args.seed)
    sim.run()