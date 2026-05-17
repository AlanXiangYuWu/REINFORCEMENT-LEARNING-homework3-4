"""Generate figures/architecture.png — a self-contained training-pipeline diagram."""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

os.makedirs("figures", exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 5.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 5.6); ax.set_aspect("equal"); ax.axis("off")


def box(x, y, w, h, text, fc="#e8f0fe", ec="#1f77b4", fontsize=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize)


def arrow(x1, y1, x2, y2, label=""):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>", mutation_scale=14,
                                 color="grey", lw=1.0))
    if label:
        ax.text((x1 + x2) / 2 + 0.1, (y1 + y2) / 2 + 0.08, label,
                fontsize=8, color="grey")


# Top: PPO + MLP on GPU
box(3.0, 4.4, 4.0, 0.9,
    "PPO (Stable-Baselines3)\nMlpPolicy: π(a|s) + V(s) — GPU",
    fc="#fde8e8", ec="#c0392b")

# Middle: 8 envs
for i in range(8):
    box(0.4 + i * 1.15, 2.6, 1.0, 0.7, f"Env {i}\nDoorKey", fontsize=8)
box(0.2, 2.4, 9.6, 1.1, "", fc="none", ec="#1f77b4")
ax.text(5.0, 3.45, "SubprocVecEnv (n=8) + FlatObsWrapper",
        ha="center", fontsize=9, color="#1f77b4")

# Bottom: callbacks + artefacts
box(0.3, 0.9, 2.5, 0.8, "CheckpointCallback\nevery 500k steps",
    fc="#eef7ec", ec="#2e7d32")
box(3.7, 0.9, 2.5, 0.8, "EvalCallback\n10 ep / 100k steps",
    fc="#eef7ec", ec="#2e7d32")
box(7.1, 0.9, 2.5, 0.8, "TensorBoard\nlogs/", fc="#eef7ec", ec="#2e7d32")

box(0.3, 0.0, 2.5, 0.6, "checkpoints/*.zip", fc="#f4f4f4", ec="grey",
    fontsize=8)
box(3.7, 0.0, 2.5, 0.6, "best_model/*.zip", fc="#f4f4f4", ec="grey",
    fontsize=8)
box(7.1, 0.0, 2.5, 0.6, "figures/*.png (plot.py)", fc="#f4f4f4", ec="grey",
    fontsize=8)

# Arrows
arrow(5.0, 4.4, 5.0, 3.55, "actions")
arrow(5.0, 2.45, 5.0, 1.75, "obs, reward")
arrow(1.55, 1.7, 1.55, 0.65)
arrow(4.95, 1.7, 4.95, 0.65)
arrow(8.35, 1.7, 8.35, 0.65)
arrow(7.0, 4.85, 8.6, 1.75, "scalars")

fig.tight_layout()
fig.savefig("figures/architecture.png", dpi=150, bbox_inches="tight")
print("wrote figures/architecture.png")
