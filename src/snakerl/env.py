"""Headless Snake as a Gymnasium environment.

Actions are relative to the current heading: 0 = straight, 1 = turn right, 2 = turn left.
Observation: 11 binary features (danger straight/right/left, heading one-hot, food direction).
Reward: +10 for food, -10 for death, 0 otherwise. An episode is truncated when the snake goes
100 * length steps without eating (prevents endless loops).
"""
from __future__ import annotations

from collections import deque
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left (x, y)


class SnakeEnv(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": ["rgb_array"], "render_fps": 12}

    def __init__(self, size: int = 10, render_mode: str | None = None):
        self.size = size
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(11,), dtype=np.float32)

    # ---- core ----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        c = self.size // 2
        self.heading = 1
        self.snake = deque([(c, c), (c - 1, c), (c - 2, c)])  # head first
        self.body = set(self.snake)
        self.score = 0
        self.steps = 0
        self.since_food = 0
        self.dead = False
        self._place_food()
        return self._obs(), {"score": 0}

    def _place_food(self):
        free = [(x, y) for x in range(self.size) for y in range(self.size) if (x, y) not in self.body]
        self.food = None if not free else free[self.np_random.integers(len(free))]

    def _collides(self, p, growing: bool) -> bool:
        x, y = p
        if not (0 <= x < self.size and 0 <= y < self.size):
            return True
        tail = self.snake[-1]
        return p in self.body and (growing or p != tail)

    def step(self, action):
        assert self.action_space.contains(action), action
        self.heading = (self.heading + (0, 1, -1)[int(action)]) % 4
        dx, dy = DIRS[self.heading]
        hx, hy = self.snake[0]
        new = (hx + dx, hy + dy)
        eat = new == self.food
        self.steps += 1
        self.since_food += 1
        if self._collides(new, eat):
            self.dead = True
            return self._obs(), -10.0, True, False, self._info("collision")
        if not eat:
            self.body.discard(self.snake.pop())
        self.snake.appendleft(new)
        self.body.add(new)
        reward = 0.0
        if eat:
            self.score += 1
            self.since_food = 0
            reward = 10.0
            self._place_food()
            if self.food is None:
                return self._obs(), reward, True, False, self._info("win")
        truncated = self.since_food >= 100 * len(self.snake)
        return self._obs(), reward, False, truncated, self._info("starved" if truncated else None)

    def _info(self, end):
        return {"score": self.score, "length": len(self.snake), "end": end}

    def _obs(self):
        hx, hy = self.snake[0]
        h = self.heading
        def danger(d):
            dx, dy = DIRS[d]
            return self._collides((hx + dx, hy + dy), False)
        fx, fy = self.food if self.food else (hx, hy)
        o = [danger(h), danger((h + 1) % 4), danger((h - 1) % 4),
             h == 0, h == 1, h == 2, h == 3,
             fy < hy, fx > hx, fy > hy, fx < hx]
        return np.array(o, dtype=np.float32)

    # ---- rendering ----
    def render(self, cell: int = 24):
        s = self.size
        img = np.full((s * cell, s * cell, 3), 18, dtype=np.uint8)
        def fill(p, col):
            x, y = p
            img[y * cell + 1:(y + 1) * cell - 1, x * cell + 1:(x + 1) * cell - 1] = col
        if self.food:
            fill(self.food, (255, 60, 120))
        n = len(self.snake)
        for i, p in enumerate(self.snake):
            k = 1 - 0.6 * i / max(n - 1, 1)
            fill(p, (0, int(230 * k), int(255 * k)))
        if self.dead:
            fill(self.snake[0], (255, 200, 0))
        return img
