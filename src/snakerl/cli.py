"""Command line: snakerl train | evaluate | gif."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .dqn import Config, load_policy, train
from .evaluate import evaluate
from .policies import greedy_policy, random_policy

EVAL_SEEDS = range(1000, 1200)  # never used during training (training seed 0..)


def main(argv=None):
    p = argparse.ArgumentParser(prog="snakerl")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--steps", type=int, default=Config.total_steps)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default="runs/seed0")
    e = sub.add_parser("evaluate")
    e.add_argument("--runs", nargs="+", default=["runs/seed0"])
    e.add_argument("--episodes", type=int, default=200)
    e.add_argument("--out", default="results/eval.json")
    g = sub.add_parser("gif")
    g.add_argument("--run", default="runs/seed0")
    g.add_argument("--out", default="docs/training.gif")
    a = p.parse_args(argv)

    if a.cmd == "train":
        cps = tuple(sorted({0, 5_000, 25_000, 100_000, a.steps} & set(range(a.steps + 1))))
        train(Config(total_steps=a.steps, seed=a.seed, checkpoints=cps), a.out)
    elif a.cmd == "evaluate":
        seeds = list(EVAL_SEEDS)[: a.episodes]
        res = {"random": evaluate(random_policy(np.random.default_rng(0)), seeds),
               "greedy heuristic": evaluate(greedy_policy, seeds)}
        for run in a.runs:
            for c in sorted(Path(run).glob("step_*.pt")):
                res[f"DQN {Path(run).name} step {int(c.stem.split('_')[1]):,}"] = evaluate(
                    load_policy(c), seeds)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
        for k, v in res.items():
            print(f"{k:<32} mean {v['mean']:>6} CI95 {v['ci95']}  max {v['max']:>3}  {v['ends']}")
    elif a.cmd == "gif":
        from .gif import make_gif
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        print(make_gif(a.run, a.out), "frames ->", a.out)


if __name__ == "__main__":
    main()
