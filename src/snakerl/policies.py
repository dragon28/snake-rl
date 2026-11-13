"""Baseline policies, all using the same 11-feature observation as the DQN."""
from __future__ import annotations

import numpy as np


def random_policy(rng: np.random.Generator):
    return lambda obs: int(rng.integers(3))


def greedy_policy(obs) -> int:
    """Turn toward the food if that move is safe, otherwise take any safe move."""
    danger = obs[0:3]  # straight, right, left
    h = int(np.argmax(obs[3:7]))  # 0 up,1 right,2 down,3 left
    food = obs[7:11]  # food up, right, down, left
    # absolute direction for each relative action
    abs_dir = [h, (h + 1) % 4, (h - 1) % 4]
    for a in range(3):
        if not danger[a] and food[abs_dir[a]]:
            return a
    for a in range(3):
        if not danger[a]:
            return a
    return 0
