"""Build the training-progress GIF from real checkpoints (fixed seed, not cherry-picked)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .dqn import load_policy
from .evaluate import run_episode


def _label(frame: np.ndarray, text: str) -> Image.Image:
    img = Image.fromarray(frame)
    bar = Image.new("RGB", (img.width, 22), (10, 10, 10))
    ImageDraw.Draw(bar).text((6, 5), text, fill=(230, 230, 230))
    out = Image.new("RGB", (img.width, img.height + 22))
    out.paste(bar, (0, 0))
    out.paste(img, (0, 22))
    return out


def make_gif(ckpt_dir: str | Path, out: str | Path, seed: int = 1000, size: int = 10,
             max_frames: int = 250, fps: int = 15):
    ckpts = sorted(Path(ckpt_dir).glob("step_*.pt"))
    frames = []
    for c in ckpts:
        step = int(c.stem.split("_")[1])
        res, imgs = run_episode(load_policy(c), seed, size, frames=True)
        imgs = imgs[:max_frames]
        for i, f in enumerate(imgs):
            frames.append(_label(f, f"after {step:,} steps | food {res['score']}"
                                 if i == len(imgs) - 1 else f"after {step:,} training steps"))
        frames += [frames[-1]] * fps  # pause on the final frame of each stage
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=1000 // fps, loop=0,
                   optimize=True)
    return len(frames)
