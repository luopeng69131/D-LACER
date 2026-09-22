#!/usr/bin/env python3
"""Convert a cleaned Starlink trace pickle into paper-aligned windows."""

from __future__ import annotations

import argparse

from dlacer.data import load_starnet_pickle, save_windowed_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to dataset_tp_sat.pkl")
    parser.add_argument("--output", required=True, help="Destination .npz archive")
    parser.add_argument("--name", required=True, help="Dataset label, e.g. chi")
    parser.add_argument("--step-len", required=True, type=int, help="Window stride")
    parser.add_argument("--input-len", type=int, default=75)
    parser.add_argument("--output-len", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_starnet_pickle(
        args.input,
        name=args.name,
        step_len=args.step_len,
        input_len=args.input_len,
        output_len=args.output_len,
    )
    output = save_windowed_data(data, args.output)
    print(
        f"Saved {len(data.X)} windows for {data.name} to {output} "
        f"(X={data.X.shape}, y={data.y.shape})."
    )


if __name__ == "__main__":
    main()
