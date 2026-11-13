"""Evaluate policies on fixed seeds, with exploration off."""
from __future__ import annotations

import numpy as np

from .env import SnakeEnv


def run_episode(policy, seed: int, size: int = 10, frames: bool = False, max_steps: int = 5_000):
    env = SnakeEnv(size)
    obs, _ = env.reset(seed=seed)
    imgs = [env.render()] if frames else []
    info = {"score": 0, "end": None}
    for _ in range(max_steps):
        obs, _, term, trunc, info = env.step(policy(obs))
        if frames:
            imgs.append(env.render())
        if term or trunc:
            break
    return {"score": info["score"], "steps": env.steps, "end": info.get("end")}, imgs


def evaluate(policy, seeds, size: int = 10) -> dict:
    res = [run_episode(policy, s, size)[0] for s in seeds]
    scores = np.array([r["score"] for r in res], float)
    rng = np.random.default_rng(42)
    boots = [rng.choice(scores, len(scores)).mean() for _ in range(1000)]
    ends = {}
    for r in res:
        ends[r["end"]] = ends.get(r["end"], 0) + 1
    return {
        "episodes": len(res),
        "mean": round(float(scores.mean()), 2),
        "ci95": [round(float(np.percentile(boots, 2.5)), 2), round(float(np.percentile(boots, 97.5)), 2)],
        "median": float(np.median(scores)),
        "max": int(scores.max()),
        "mean_steps": round(float(np.mean([r["steps"] for r in res])), 1),
        "ends": ends,
    }
