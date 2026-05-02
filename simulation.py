"""
simulation.py
===============

Grid-based simulation for a longevity society: loads YAML, runs agents, and
writes optional artifacts (summary, CSV, replay JSON, PPM frames, MP4 if ffmpeg
is available). Optional Ollama LLM use is off by default (``llm.enabled: false``).
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml

from agent import Agent, Place


@dataclass
class Event:
    """Represents a time-triggered event."""

    name: str
    start_step: int
    intensity: float
    radius: int
    center_x: int
    center_y: int

    def is_active(self, step: int) -> bool:
        return step >= self.start_step


class Simulation:
    """Loads configuration, runs agents step by step, and saves outputs."""

    def __init__(self, config_path: str, seed: Optional[int] = None) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            self.config: dict = yaml.safe_load(f)

        log_file = self.config.get("logging", {}).get("log_file", "simulation.log")
        logging.basicConfig(
            filename=log_file,
            level=getattr(logging, self.config.get("logging", {}).get("level", "INFO")),
            filemode="w",
            format="%(asctime)s %(levelname)s:%(message)s",
        )
        logging.info("Initialising simulation")

        self.rng = random.Random(seed)
        self.seed = seed

        llm_cfg = self.config.get("llm", {})
        self.llm_enabled: bool = bool(llm_cfg.get("enabled", False))
        self._llm_disabled_runtime: bool = False
        self._llm_failure_count: int = 0

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

        num_agents = self.config.get("agents", {}).get("num_agents", 10)
        self.agents: List[Agent] = []
        for i in range(num_agents):
            self.agents.append(Agent(agent_id=i, config=self.config, places=self.places, rng=self.rng))

        self.visit_events: List[Tuple[int, int, str]] = []
        self.replay_frames: List[Dict[str, Any]] = []

        viz = self.config.get("visualization", {})
        self._output_dir = viz.get("output_dir", "output_longevity")
        self._save_frames = bool(viz.get("save_frames", False))
        self._frame_interval = max(1, int(viz.get("frame_interval", 1)))
        self._cell_px = max(1, int(viz.get("cell_pixels", 8)))
        self._fps = max(1, int(viz.get("fps", 10)))

    @property
    def llm_effective(self) -> bool:
        return self.llm_enabled and not self._llm_disabled_runtime

    def register_llm_failure(self) -> None:
        self._llm_failure_count += 1
        lim = self.config.get("llm", {}).get("failure_limit", 5)
        try:
            lim_i = int(lim)
        except (TypeError, ValueError):
            lim_i = 5
        if self._llm_failure_count >= lim_i:
            logging.warning(
                "LLM failures reached limit (%s); switching to heuristic-only for remaining steps",
                lim_i,
            )
            self._llm_disabled_runtime = True

    def record_visit(self, step: int, agent_id: int, place_name: str) -> None:
        self.visit_events.append((step, agent_id, place_name))

    def run(self) -> None:
        duration = self.config["simulation"]["duration"]
        logging.info(f"Running simulation for {duration} steps with {len(self.agents)} agents")
        start_time = time.time()

        os.makedirs(self._output_dir, exist_ok=True)
        frames_dir = os.path.join(self._output_dir, "frames")
        if self._save_frames:
            os.makedirs(frames_dir, exist_ok=True)

        global_visits: Dict[str, int] = {name: 0 for name in self.places.keys()}
        ppm_paths: List[str] = []
        ppm_seq = 0

        half_space = self.config["simulation"]["half_space_size"]

        for step in range(duration):
            frame_agents: List[Dict[str, Any]] = []
            for agent in self.agents:
                agent.step(self, step)
                frame_agents.append(
                    {
                        "id": agent.id,
                        "x": agent.x,
                        "y": agent.y,
                        "place": agent.current_place,
                    }
                )

            self.replay_frames.append({"step": step, "agents": frame_agents})

            if self._save_frames and step % self._frame_interval == 0:
                fname = os.path.join(frames_dir, f"frame_{ppm_seq:06d}.ppm")
                ppm_seq += 1
                self._write_ppm(fname, frame_agents, half_space)
                ppm_paths.append(fname)

        for agent in self.agents:
            for name, count in agent.visit_counts.items():
                global_visits[name] += count

        elapsed = time.time() - start_time
        logging.info("Simulation completed in %.2fs", elapsed)

        self._write_artifacts(
            duration=duration,
            elapsed=elapsed,
            global_visits=global_visits,
            half_space=half_space,
            ppm_paths=ppm_paths,
            frames_dir=frames_dir if self._save_frames else "",
        )

        print("Simulation complete.")
        print(f"Duration: {duration} steps, agents: {len(self.agents)}, time: {elapsed:.2f}s")
        print("Total visits per place:")
        for name, total in global_visits.items():
            print(f"  {name}: {total}")
        print(f"Outputs written under: {os.path.abspath(self._output_dir)}")

    def _place_meta(self) -> List[Dict[str, Any]]:
        out = []
        for p in self.config.get("places", []):
            out.append(
                {
                    "name": p["name"],
                    "center_x": p.get("center_x", 0),
                    "center_y": p.get("center_y", 0),
                    "half_size": p.get("half_size", self.config["simulation"]["half_place_size"]),
                }
            )
        return out

    def _write_artifacts(
        self,
        duration: int,
        elapsed: float,
        global_visits: Dict[str, int],
        half_space: int,
        ppm_paths: List[str],
        frames_dir: str,
    ) -> None:
        summary_path = os.path.join(self._output_dir, "summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"duration_steps: {duration}\n")
            f.write(f"agents: {len(self.agents)}\n")
            f.write(f"elapsed_seconds: {elapsed:.4f}\n")
            f.write(f"seed: {self.seed}\n")
            f.write(f"llm.enabled_config: {self.llm_enabled}\n")
            f.write(f"llm.still_enabled_after_run: {not self._llm_disabled_runtime}\n")
            f.write(f"llm.failures: {self._llm_failure_count}\n")
            f.write(f"llm.disabled_runtime: {self._llm_disabled_runtime}\n")
            f.write("total_visits_per_place:\n")
            for name, total in sorted(global_visits.items()):
                f.write(f"  {name}: {total}\n")

        visits_path = os.path.join(self._output_dir, "visits.csv")
        with open(visits_path, "w", encoding="utf-8") as f:
            f.write("step,agent_id,place\n")
            for step, aid, pname in self.visit_events:
                f.write(f"{step},{aid},{pname}\n")

        replay_path = os.path.join(self._output_dir, "replay.json")
        replay_doc = {
            "meta": {
                "duration": duration,
                "half_space": half_space,
                "seed": self.seed,
                "places_version": 1,
            },
            "places": self._place_meta(),
            "frames": self.replay_frames,
        }
        with open(replay_path, "w", encoding="utf-8") as f:
            json.dump(replay_doc, f, indent=2)

        viewer_src = os.path.join(os.path.dirname(__file__), "replay_viewer.html")
        viewer_dst = os.path.join(self._output_dir, "replay_viewer.html")
        if os.path.isfile(viewer_src):
            shutil.copy2(viewer_src, viewer_dst)

        mp4_path = os.path.join(self._output_dir, "simulation.mp4")
        if ppm_paths and shutil.which("ffmpeg"):
            pattern = os.path.join(frames_dir, "frame_%06d.ppm")
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                str(self._fps),
                "-i",
                pattern,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                mp4_path,
            ]
            try:
                subprocess.run(cmd, check=True)
                logging.info("Wrote %s", mp4_path)
            except subprocess.CalledProcessError as e:
                logging.warning("ffmpeg failed: %s", e)
        elif ppm_paths:
            logging.info("ffmpeg not found; PPM frames kept under %s", frames_dir)

    def _write_ppm(
        self,
        path: str,
        agents_snapshot: List[Dict[str, Any]],
        half_space: int,
    ) -> None:
        """Write a binary PPM (P6) bird's-eye view of the grid."""
        cell = self._cell_px
        grid = 2 * half_space + 1
        w = grid * cell
        h = grid * cell
        bg_r, bg_g, bg_b = 20, 24, 32

        buf = bytearray([bg_r, bg_g, bg_b] * (w * h))

        def set_px(px: int, py: int, r: int, g: int, b: int) -> None:
            if 0 <= px < w and 0 <= py < h:
                i = (py * w + px) * 3
                buf[i : i + 3] = bytes([r, g, b])

        def fill_cell(cx: int, cy: int, r: int, g: int, b: int, alpha: float = 0.25) -> None:
            for dy in range(cell):
                for dx in range(cell):
                    px = cx * cell + dx
                    py = (grid - 1 - cy) * cell + dy
                    bi = (py * w + px) * 3
                    if 0 <= px < w and 0 <= py < h:
                        br, bg, bb = buf[bi], buf[bi + 1], buf[bi + 2]
                        nr = int(br * (1 - alpha) + r * alpha)
                        ng = int(bg * (1 - alpha) + g * alpha)
                        nb = int(bb * (1 - alpha) + b * alpha)
                        buf[bi : bi + 3] = bytes([nr, ng, nb])

        palette = [
            (220, 90, 90),
            (90, 200, 120),
            (100, 160, 255),
            (240, 200, 80),
            (200, 120, 220),
            (80, 220, 200),
            (255, 140, 80),
            (180, 180, 255),
        ]

        for pname, place in self.places.items():
            cx, cy = place.center
            hs = place.half_size
            for gx in range(cx - hs, cx + hs + 1):
                for gy in range(cy - hs, cy + hs + 1):
                    if -half_space <= gx <= half_space and -half_space <= gy <= half_space:
                        ix = gx + half_space
                        iy = gy + half_space
                        fill_cell(ix, iy, 60, 70, 100, alpha=0.35)

        for a in agents_snapshot:
            gx = a["x"] + half_space
            gy = a["y"] + half_space
            if 0 <= gx < grid and 0 <= gy < grid:
                pr, pg, pb = palette[a["id"] % len(palette)]
                fill_cell(gx, gy, pr, pg, pb, alpha=0.85)

        header = f"P6\n{w} {h}\n255\n".encode("ascii")
        with open(path, "wb") as f:
            f.write(header)
            f.write(buf)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run longevity society simulation")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    sim = Simulation(args.config, seed=args.seed)
    sim.run()
