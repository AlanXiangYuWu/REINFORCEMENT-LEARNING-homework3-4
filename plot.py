"""Generate training-curve figures from TensorBoard event files."""
import argparse
import glob
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalar(event_dir: str, tag: str):
    files = sorted(glob.glob(os.path.join(event_dir, "**", "events.out.tfevents.*"),
                             recursive=True))
    steps, values = [], []
    for f in files:
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        if tag not in ea.Tags()["scalars"]:
            continue
        for e in ea.Scalars(tag):
            steps.append(e.step)
            values.append(e.value)
    if not steps:
        return np.array([]), np.array([])
    order = np.argsort(steps)
    return np.array(steps)[order], np.array(values)[order]


def smooth(y: np.ndarray, w: int = 11) -> np.ndarray:
    if len(y) < w or w < 2:
        return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode="same")


def plot_one(log_dir, tag, out_path, ylabel, title, phases=None):
    steps, vals = load_scalar(log_dir, tag)
    if len(steps) == 0:
        print(f"[skip] no data for {tag}")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, vals, alpha=0.25, color="C0", label="raw")
    ax.plot(steps, smooth(vals, 21), color="C0", linewidth=2, label="smoothed")
    if phases:
        for x, name in phases:
            if x <= steps[-1]:
                ax.axvline(x, color="grey", linestyle="--", alpha=0.5)
                ax.text(x, ax.get_ylim()[1] * 0.95, name, rotation=90,
                        va="top", fontsize=8, color="grey")
    ax.set_xlabel("env_steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[ok] wrote {out_path}  ({len(steps)} points, last step={steps[-1]})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="./logs/")
    parser.add_argument("--out-dir", default="./figures/")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    phases = [
        (200_000,    "P1"),
        (1_000_000,  "P2"),
        (3_000_000,  "P3"),
        (10_000_000, "P4"),
    ]
    plot_one(args.log_dir, "rollout/ep_rew_mean",
             os.path.join(args.out_dir, "training_curve.png"),
             "episode reward (mean)",
             "PPO on MiniGrid-DoorKey-8x8 — episode reward",
             phases=phases)
    plot_one(args.log_dir, "rollout/ep_len_mean",
             os.path.join(args.out_dir, "episode_length.png"),
             "episode length (mean)",
             "PPO on MiniGrid-DoorKey-8x8 — episode length",
             phases=phases)
    plot_one(args.log_dir, "train/entropy_loss",
             os.path.join(args.out_dir, "entropy.png"),
             "entropy loss",
             "PPO on MiniGrid-DoorKey-8x8 — policy entropy")
    plot_one(args.log_dir, "train/approx_kl",
             os.path.join(args.out_dir, "kl.png"),
             "approx KL",
             "PPO on MiniGrid-DoorKey-8x8 — approx KL")
    plot_one(args.log_dir, "eval/mean_reward",
             os.path.join(args.out_dir, "eval_reward.png"),
             "eval mean reward",
             "PPO on MiniGrid-DoorKey-8x8 — eval reward")


if __name__ == "__main__":
    main()
