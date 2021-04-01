#!/usr/bin/env python

# Copyright (c) 2021 Kemal Kurniawan


from pathlib import Path
import argparse
import json
import random


def generate():
    return {
        "lr": 10 ** random.uniform(-6, -4),  # [1e-6 .. 1e-4)
        "temperature": random.uniform(1, 10),  # [1 .. 10)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate hyopt configs for Tackstrom et al.'s AAST"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        type=Path,
        help="where to store the generated config files",
    )
    parser.add_argument(
        "--n-configs", "-n", required=True, type=int, help="number of config files to generate"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="whether to overwrite output dir if exists"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=args.overwrite)

    for k in range(args.n_configs):
        with open(args.output_dir / f"config_{k}.json", "w", encoding="utf8") as f:
            json.dump(generate(), f, indent=2, sort_keys=True)
