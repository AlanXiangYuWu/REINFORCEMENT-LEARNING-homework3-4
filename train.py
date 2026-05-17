"""PPO on MiniGrid-DoorKey-8x8-v0 — homework 3 & 4."""
import os
import argparse

import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

DEFAULT_ENV_ID = "MiniGrid-DoorKey-16x16-v0"


def make_env(env_id: str, seed: int):
    def _init():
        env = gym.make(env_id)
        env = FlatObsWrapper(env)
        env.reset(seed=seed)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default=DEFAULT_ENV_ID)
    parser.add_argument("--total-steps", type=int, default=20_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--log-dir", type=str, default="./logs/")
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints/")
    parser.add_argument("--best-dir", type=str, default="./best_model/")
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.best_dir, exist_ok=True)

    print(f"[env] {args.env_id}  n_envs={args.n_envs}  total_steps={args.total_steps:,}")

    vec_env = SubprocVecEnv([make_env(args.env_id, i) for i in range(args.n_envs)])
    vec_env = VecMonitor(vec_env)

    eval_env = SubprocVecEnv([make_env(args.env_id, 1000 + i) for i in range(2)])
    eval_env = VecMonitor(eval_env)

    if args.resume and os.path.isfile(args.resume):
        print(f"[resume] loading {args.resume}")
        model = PPO.load(args.resume, env=vec_env, device="cuda",
                         tensorboard_log=args.log_dir)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=2.5e-4,
            n_steps=128,
            batch_size=256,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=args.log_dir,
            device="cuda",
            verbose=1,
        )

    ckpt_cb = CheckpointCallback(
        save_freq=max(500_000 // args.n_envs, 1),
        save_path=args.ckpt_dir,
        name_prefix="ppo",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.best_dir,
        log_path=args.log_dir,
        eval_freq=max(100_000 // args.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=args.total_steps,
        callback=[ckpt_cb, eval_cb],
        reset_num_timesteps=not bool(args.resume),
        progress_bar=False,
    )
    model.save(os.path.join(args.ckpt_dir, "ppo_final"))


if __name__ == "__main__":
    main()
