"""A small, hand-written Double DQN (replay buffer + target network)."""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .env import SnakeEnv


@dataclass
class Config:
    size: int = 10
    total_steps: int = 300_000
    buffer: int = 100_000
    batch: int = 256
    gamma: float = 0.95
    lr: float = 5e-4
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_steps: int = 100_000
    target_every: int = 1_000
    learn_start: int = 1_000
    hidden: int = 128
    seed: int = 0
    checkpoints: tuple = (0, 5_000, 25_000, 100_000, 300_000)


def make_net(hidden: int = 128) -> nn.Module:
    return nn.Sequential(nn.Linear(11, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, 3))


def net_policy(net: nn.Module):
    def act(obs) -> int:
        with torch.no_grad():
            return int(net(torch.as_tensor(obs).unsqueeze(0)).argmax())
    return act


def load_policy(path: str | Path):
    blob = torch.load(path, weights_only=True)
    net = make_net(blob["hidden"])
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return net_policy(net)


def train(cfg: Config, out: str | Path, log=print) -> list[dict]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(1)
    env = SnakeEnv(cfg.size)
    q, tgt = make_net(cfg.hidden), make_net(cfg.hidden)
    tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=cfg.lr)
    S = np.zeros((cfg.buffer, 11), np.float32)
    S2 = np.zeros((cfg.buffer, 11), np.float32)
    A = np.zeros(cfg.buffer, np.int64)
    R = np.zeros(cfg.buffer, np.float32)
    D = np.zeros(cfg.buffer, np.float32)
    n = ptr = 0
    history, ep_scores = [], []
    obs, _ = env.reset(seed=cfg.seed)
    ckpts = set(cfg.checkpoints)

    def save(step):
        torch.save({"state_dict": q.state_dict(), "hidden": cfg.hidden, "step": step},
                   out / f"step_{step:07d}.pt")

    for step in range(cfg.total_steps + 1):
        if step in ckpts:
            save(step)
        if step == cfg.total_steps:
            break
        eps = max(cfg.eps_end, cfg.eps_start - (cfg.eps_start - cfg.eps_end) * step / cfg.eps_decay_steps)
        if random.random() < eps:
            a = random.randrange(3)
        else:
            with torch.no_grad():
                a = int(q(torch.as_tensor(obs).unsqueeze(0)).argmax())
        obs2, r, term, trunc, info = env.step(a)
        S[ptr], A[ptr], R[ptr], S2[ptr], D[ptr] = obs, a, r, obs2, float(term)  # truncation bootstraps
        ptr = (ptr + 1) % cfg.buffer
        n = min(n + 1, cfg.buffer)
        obs = obs2
        if term or trunc:
            ep_scores.append(info["score"])
            obs, _ = env.reset()
        if n >= cfg.learn_start:
            idx = np.random.randint(0, n, cfg.batch)
            s, a_, r_, s2, d = (torch.as_tensor(x[idx]) for x in (S, A, R, S2, D))
            with torch.no_grad():
                a2 = q(s2).argmax(1, keepdim=True)
                y = r_ + cfg.gamma * (1 - d) * tgt(s2).gather(1, a2).squeeze(1)
            pred = q(s).gather(1, a_.unsqueeze(1)).squeeze(1)
            loss = nn.functional.smooth_l1_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10)
            opt.step()
        if step % cfg.target_every == 0:
            tgt.load_state_dict(q.state_dict())
        if step and step % 5_000 == 0:
            recent = ep_scores[-100:] or [0]
            history.append({"step": step, "episodes": len(ep_scores), "eps": round(eps, 3),
                            "mean_score_last100": float(np.mean(recent))})
            log(f"step {step:>7}  episodes {len(ep_scores):>5}  eps {eps:.2f}  "
                f"mean score (last 100) {np.mean(recent):.2f}")
    (out / "history.json").write_text(json.dumps(history, indent=1))
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    return history
