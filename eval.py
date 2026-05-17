"""Roll out the trained policy on DoorKey-8x8 and save frame snapshots."""
import argparse
import os

import gymnasium as gym
import matplotlib.pyplot as plt
from minigrid.wrappers import FlatObsWrapper, RGBImgObsWrapper
from stable_baselines3 import PPO

DEFAULT_ENV_ID = "MiniGrid-DoorKey-16x16-v0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID)
    parser.add_argument("--model", default="./best_model/best_model.zip")
    parser.add_argument("--out-dir", default="./figures/")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    raw_env = gym.make(args.env_id, render_mode="rgb_array")
    model_env = FlatObsWrapper(gym.make(args.env_id))
    model = PPO.load(args.model, device="cpu")

    for ep in range(args.episodes):
        obs, _ = raw_env.reset(seed=ep * 7)
        flat_env = FlatObsWrapper(raw_env)
        obs_flat = flat_env.observation(obs)
        frames = [raw_env.render()]
        done = False
        total_r = 0.0
        steps = 0
        while not done and steps < 200:
            action, _ = model.predict(obs_flat, deterministic=True)
            obs, r, term, trunc, _ = raw_env.step(int(action))
            obs_flat = flat_env.observation(obs)
            frames.append(raw_env.render())
            total_r += r
            steps += 1
            done = term or trunc
        print(f"[ep {ep}] reward={total_r:.3f}  length={steps}")

        n = min(6, len(frames))
        idx = [int(i * (len(frames) - 1) / (n - 1)) for i in range(n)]
        fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.6))
        for k, j in enumerate(idx):
            axes[k].imshow(frames[j])
            axes[k].axis("off")
            axes[k].set_title(f"t={j}")
        fig.suptitle(f"Episode {ep} — return {total_r:.2f}, length {steps}")
        fig.tight_layout()
        out = os.path.join(args.out_dir, f"rollout_ep{ep}.png")
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
