"""Overlay TensorBoard scalars from the four MiniGrid runs."""
import glob, os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS = [
    ("DoorKey-8x8",       "archive_doorkey/logs",          "#1f77b4"),
    ("MultiRoom-N2-S4",   "archive_multiroom_n2_s4/logs",  "#2ca02c"),
    ("DoorKey-16x16",     "logs",                          "#ff7f0e"),
    ("MultiRoom-N4-S5",   "archive_multiroom_n4_s5/logs",  "#d62728"),
]

OUT = "figures"
os.makedirs(OUT, exist_ok=True)


def load(log_dir, tag):
    f = sorted(glob.glob(os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True))
    if not f:
        return np.array([]), np.array([])
    ea = EventAccumulator(f[0], size_guidance={"scalars": 0}); ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return np.array([]), np.array([])
    s = ea.Scalars(tag)
    return np.array([e.step for e in s]), np.array([e.value for e in s])


def smooth(y, w):
    if len(y) < w or w < 2: return y
    return np.convolve(y, np.ones(w)/w, mode="same")


def overlay(tag, ylabel, title, fname, ylim=None, logx=False, smooth_w=51):
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    for name, ldir, color in RUNS:
        x, y = load(ldir, tag)
        if len(x) == 0:
            continue
        ys = smooth(y, smooth_w)
        ax.plot(x, ys, color=color, linewidth=2, label=name)
        ax.plot(x, y, color=color, alpha=0.12, linewidth=0.8)
    if logx:
        ax.set_xscale("symlog", linthresh=10_000)
    ax.set_xlabel("env_steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim: ax.set_ylim(ylim)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out = os.path.join(OUT, fname)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


overlay("rollout/ep_rew_mean", "episode reward (mean)",
        "Episode reward across four MiniGrid tasks (smoothed)",
        "cmp_reward.png", ylim=(-0.02, 1.0))
overlay("rollout/ep_len_mean", "episode length (mean)",
        "Episode length across four MiniGrid tasks (smoothed)",
        "cmp_length.png")
overlay("train/entropy_loss", "entropy loss",
        "Policy entropy loss across four MiniGrid tasks (smoothed)",
        "cmp_entropy.png")

# Bar chart of final reward
fig, ax = plt.subplots(figsize=(7.5, 3.6))
names, finals, maxes, colors = [], [], [], []
for name, ldir, c in RUNS:
    _, y = load(ldir, "rollout/ep_rew_mean")
    if len(y):
        names.append(name); colors.append(c)
        finals.append(float(np.mean(y[-50:]) if len(y) >= 50 else y[-1]))
        maxes.append(float(y.max()))
xs = np.arange(len(names))
ax.bar(xs - 0.18, finals, width=0.36, color=colors, label="last-50-iters mean", alpha=0.95)
ax.bar(xs + 0.18, maxes,  width=0.36, color=colors, label="max during training", alpha=0.55, hatch="//")
ax.set_xticks(xs); ax.set_xticklabels(names, rotation=15)
ax.set_ylabel("episode reward")
ax.set_title("Final vs. peak training reward by environment")
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "cmp_final_bar.png"), dpi=150)
plt.close(fig)
print("wrote", os.path.join(OUT, "cmp_final_bar.png"))

# Eval reward overlay
overlay("eval/mean_reward", "eval mean reward",
        "Deterministic eval reward across four MiniGrid tasks",
        "cmp_eval.png", ylim=(-0.02, 1.0), smooth_w=5)
