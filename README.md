# PPO on MiniGrid — Homework 3 & 4

Reinforcement-learning coursework: train PPO on MiniGrid, draw the reward
vs `env_steps` curve, analyse the behavioural phases, and deliver a
5-page report. This repo contains the training pipeline plus four
experiments that together map out where PPO + SB3 defaults work on
MiniGrid and where they break.

Final deliverable: [`report.pdf`](report.pdf) (5 pages).

## Results at a glance

| Environment       | env_steps | Wall-clock | Final reward | Outcome      |
| ----------------- | --------: | ---------: | -----------: | :----------- |
| DoorKey-8x8       |    4.07 M |     15 min |    **0.975** | converged    |
| MultiRoom-N2-S4   |   17.68 M |     65 min |    **0.843** | converged    |
| DoorKey-16x16     |   20.00 M |     99 min |     0.014    | partial      |
| MultiRoom-N4-S5   |   20.00 M |     71 min |     0.000    | failed       |

All four runs use identical PPO hyperparameters (SB3 reference recipe);
the only change across runs is `--env-id`. The full discussion lives in
the report; the headline is that MiniGrid presents a *binary difficulty
cliff* under fixed hyperparameters — tasks are either solved within
~500 k env steps or PPO produces essentially zero learning at 20 M steps.

## Project layout

```
.
├── PLAN.md                              project plan (original)
├── report.typ / report.pdf              5-page Typst report
├── requirements.txt                     pinned deps
├── train.py                             PPO training entry point
├── plot.py                              per-run TB → figures
├── eval.py                              load best_model.zip, render rollouts
├── compare_plots.py                     overlay all 4 runs into one figure set
├── make_arch.py                         regenerate architecture diagram
├── figures/                             figures used by the report
│
├── logs/                                ← DoorKey-16x16 (most recent run)
├── checkpoints/                         ← only ppo_final.zip kept
├── best_model/                          ← deterministic-eval best model
│
├── archive_doorkey/                     ← DoorKey-8x8 (TB + best_model)
├── archive_multiroom_n2_s4/             ← MultiRoom-N2-S4 (full ckpts kept)
└── archive_multiroom_n4_s5/             ← MultiRoom-N4-S5 (TB + best_model;
                                           checkpoints dropped — all zero reward)
```

## Reproducing

```bash
# 1. install deps (conda env recommended)
pip install -r requirements.txt

# 2. train (default: MultiRoom-N2-S4-v0; override --env-id for others)
python train.py --env-id MiniGrid-DoorKey-8x8-v0 --total-steps 4000000 --n-envs 8

# 3. live monitor (optional)
tensorboard --logdir=logs/

# 4. generate per-run figures
python plot.py --log-dir ./logs/ --out-dir ./figures/

# 5. (optional) generate the four-run comparison figures used in the report
python compare_plots.py

# 6. render rollouts from the best model
python eval.py --model ./best_model/best_model.zip

# 7. compile the report
typst compile report.typ report.pdf
```

Resume from a checkpoint:

```bash
python train.py --resume ./checkpoints/ppo_final.zip --total-steps 30000000
```

## Hyperparameters

Held fixed across all four experiments (SB3 reference recipe; no tuning):

| | | | |
| :- | -: | :- | -: |
| Policy            | `MlpPolicy` (2×64 tanh) | Discount γ        | 0.99   |
| Learning rate     | 2.5e-4                  | GAE λ             | 0.95   |
| Workers           | 8 (`SubprocVecEnv`)     | Clip range ε      | 0.20   |
| Rollout `n_steps` | 128 per worker          | Entropy coef      | 0.01   |
| Total batch       | 1 024 transitions       | Value coef        | 0.50   |
| Minibatch         | 256                     | Max grad norm     | 0.50   |
| Update epochs     | 4                       | Device            | `cuda` |

## Reading the TensorBoard logs

Each run's TensorBoard event files are checked in so figures can be
regenerated without re-training:

```bash
tensorboard --logdir_spec \
  doorkey8:archive_doorkey/logs,\
  n2s4:archive_multiroom_n2_s4/logs,\
  doorkey16:logs,\
  n4s5:archive_multiroom_n4_s5/logs
```

## Notes

- The training scripts are deliberately small (~70 lines each). The
  point of the exercise is the *analysis*, not the implementation —
  PPO itself comes from Stable-Baselines3 2.3.2.
- `archive_multiroom_n2_s4/checkpoints/` retains all 33 intermediate
  checkpoints so the N2-S4 run can be resumed or its intermediate
  policies inspected.
- The other archives keep the TensorBoard events and `best_model.zip`
  only; intermediate `.zip` checkpoints were dropped to keep the repo
  under 230 MB.
