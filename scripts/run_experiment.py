#!/usr/bin/env python3
"""Train the public forecast bank and evaluate D-LACER on prepared data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dlacer import DLACERController, RouterConfig, build_latent_risk_groups
from dlacer.data import chronological_splits, load_windowed_archive
from dlacer.forecast import ForecastBankConfig, XGBoostForecastBank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Prepared .npz archive")
    parser.add_argument("--output", default="outputs/dlacer_result.npz")
    parser.add_argument("--budget", type=float, default=0.35)
    parser.add_argument("--estimators", type=int, default=120)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_windowed_archive(args.data)
    splits = chronological_splits(data)
    bank = XGBoostForecastBank(
        data.output_len,
        ForecastBankConfig(
            n_estimators=args.estimators,
            random_state=args.seed,
            device=args.device,
        ),
    ).fit(data.X[splits.base_train], data.y[splits.base_train])

    test = splits.test
    point = bank.predict_point(data.X[test])
    quantiles = bank.predict_quantiles(data.X[test])
    candidates = np.concatenate(
        [point[None, :, :], np.transpose(quantiles, (2, 0, 1))], axis=0
    )
    target = data.y[test]
    context = data.context[test]
    initialization_end = int(0.40 * len(test))
    evaluation_start = int(0.60 * len(test))
    feedback_delay = max(1, int(np.ceil(data.output_len / data.step_len)))

    groups = build_latent_risk_groups(
        context,
        point,
        candidates,
        target,
        initialization_end,
        RouterConfig(random_state=args.seed),
    )
    result = DLACERController().run(
        point,
        target,
        groups,
        budget=args.budget,
        initialization_end=initialization_end,
        evaluation_start=evaluation_start,
        feedback_delay=feedback_delay,
    )

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        prediction=result.prediction,
        target=target,
        point=point,
        correction_trace=result.correction_trace,
        group_names=np.asarray(groups.names, dtype=str),
        group_mask=groups.mask,
    )
    summary = {
        "dataset": data.name,
        "budget": args.budget,
        "feedback_delay": feedback_delay,
        "metrics": result.metrics,
        "group_metrics": result.group_metrics,
        "group_budget_pass": result.group_passes(args.budget),
        "output": str(output),
    }
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
