"""
main.py
=======

Entry point for running the longevity society simulation.  This script
wraps the ``Simulation`` class defined in ``simulation.py`` and
provides a simple command‐line interface.  See README.md for
instructions.
"""

import argparse

from simulation import Simulation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run longevity society simulation")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )
    args = parser.parse_args()

    sim = Simulation(args.config, seed=args.seed)
    sim.run()


if __name__ == "__main__":
    main()