# Homework 3 & 4 — PPO on MiniGrid

> 目标:用 PPO 在 MiniGrid 上训练 24+ 小时,绘制 reward vs env_steps 曲线,分析行为变化阶段,完成 5 页报告。

---

## 1. 项目目标 (Scope)

| 项 | 内容 |
|---|---|
| 算法 | PPO (Proximal Policy Optimization) |
| 框架 | Stable-Baselines3 |
| 环境 | `MiniGrid-DoorKey-8x8-v0` |
| 训练时长 | 20M env_steps (~12–20 小时 GPU) |
| 交付物 | 训练曲线图 + 5 页 PDF 报告 |
| 分数 | 5 + 5 = 10 marks |

**非目标:** 不做超参搜索、不做多算法对比、不实现 PPO 内部、不做分布式。

---

## 2. 系统架构 (Architecture)

```
                          ┌──────────────────────────────────┐
                          │          train.py                │
                          │  (主进程 / GPU)                  │
                          │                                  │
                          │   ┌────────────────────────┐     │
                          │   │   PPO (SB3)            │     │
                          │   │  ┌──────────────────┐  │     │
                          │   │  │ MlpPolicy        │  │     │
                          │   │  │ ┌──────────────┐ │  │     │
                          │   │  │ │ Actor (π)    │ │  │     │
                          │   │  │ │ Critic (V)   │ │  │     │
                          │   │  │ └──────────────┘ │  │     │
                          │   │  └──────────────────┘  │     │
                          │   └───────┬────────────────┘     │
                          │           │ actions              │
                          │           ▼                      │
                          │   ┌────────────────────────┐     │
                          │   │ SubprocVecEnv (n=8)    │     │
                          │   │ ┌────┐┌────┐ ... ┌────┐│     │
                          │   │ │Env0││Env1│     │Env7││     │
                          │   │ └────┘└────┘     └────┘│     │
                          │   │   各套 FlatObsWrapper  │     │
                          │   └────────────────────────┘     │
                          │           │ obs, reward          │
                          │           ▼                      │
                          │   ┌────────────────────────┐     │
                          │   │  Callbacks             │     │
                          │   │  - CheckpointCallback  │     │
                          │   │  - EvalCallback        │     │
                          │   └───────┬────────────────┘     │
                          └───────────┼──────────────────────┘
                                      │
                  ┌───────────────────┼───────────────────┐
                  ▼                   ▼                   ▼
          ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
          │  logs/      │    │ checkpoints/│    │ best_model/ │
          │ TB events   │    │ *.zip       │    │ *.zip       │
          └──────┬──────┘    └─────────────┘    └─────────────┘
                 │
                 ▼
          ┌─────────────┐
          │  plot.py    │ ──► figures/training_curve.png
          └─────────────┘
```

### 数据流

1. `SubprocVecEnv` 并行跑 8 个 MiniGrid 实例,每个返回 7×7×3 的 partial obs
2. `FlatObsWrapper` 把 dict obs 拍平成 1D 向量喂给 MLP
3. PPO 每 `n_steps=128` 步收集一次 rollout (8 × 128 = 1024 transitions)
4. Actor-Critic MLP 在 GPU 上做 4 个 epoch 的 minibatch 更新
5. Callbacks 周期性保存 checkpoint + 评估 + 写 TB 日志

---

## 3. 文件结构

```
homework3-4/
├── PLAN.md                  # 本文件
├── requirements.txt         # 依赖清单
├── train.py                 # ~60 行,主训练入口
├── plot.py                  # ~40 行,从 TB 日志生成报告用图
├── eval.py                  # ~30 行,加载 checkpoint 可视化跑 episode
├── logs/                    # TensorBoard 日志 (自动生成)
│   └── PPO_1/events.out...
├── checkpoints/             # 定期 checkpoint (自动生成)
│   ├── ppo_500000_steps.zip
│   └── ...
├── best_model/              # EvalCallback 保存的最优模型
│   └── best_model.zip
├── figures/                 # 报告插图
│   ├── training_curve.png
│   ├── episode_length.png
│   └── entropy.png
└── report.pdf               # 最终交付报告
```

---

## 4. 核心组件设计

### 4.1 `train.py`

**职责:** 配置环境 + PPO + callbacks,启动训练。

**关键决策:**
- `SubprocVecEnv` 而非 `DummyVecEnv`(8 个进程并行,CPU bound 任务必需)
- `FlatObsWrapper` 而非 `ImgObsWrapper`(MLP 比 CNN 收敛快,7×7 网格信息少)
- `tensorboard_log` 必开(报告画图唯一数据源)
- `CheckpointCallback(save_freq=500_000 // n_envs)`(每 500k env_steps 存一次,断电恢复)
- `EvalCallback(eval_freq=100_000 // n_envs, n_eval_episodes=10)`(独立评估,reward 曲线更平滑)

### 4.2 `plot.py`

**职责:** 读 TensorBoard event 文件,生成 3 张图。

**输出:**
- `training_curve.png` — `rollout/ep_rew_mean` vs `env_steps`(报告主图)
- `episode_length.png` — `rollout/ep_len_mean` vs `env_steps`(展示策略优化)
- `entropy.png` — `train/entropy_loss` vs `env_steps`(展示探索→利用)

### 4.3 `eval.py`

**职责:** 加载 `best_model.zip`,渲染 5 个 episode 截图供报告使用。

---

## 5. 超参数 (PPO)

```python
PPO(
    policy="MlpPolicy",
    env=vec_env,
    learning_rate=2.5e-4,
    n_steps=128,            # per env, total batch = 8*128=1024
    batch_size=256,         # minibatch
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,          # 关键:鼓励探索,避免熵塌缩
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log="./logs/",
    device="cuda",
    verbose=1,
)
```

> 这套是 SB3 PPO 在 MiniGrid 上的社区验证配置,**不打算调参**。

---

## 6. 训练计划 (Timeline)

| 阶段 | env_steps | 预期行为 | 预期 ep_rew_mean |
|---|---|---|---|
| Phase 0: 探索 | 0 – 200k | 随机游走,撞墙 | ≈ 0 |
| Phase 1: 偶然成功 | 200k – 1M | 偶然拾起钥匙 | 0 – 0.1 |
| Phase 2: 因果学习 | 1M – 3M | 学会 key→door→goal 链 | 0.1 – 0.6 |
| Phase 3: 路径优化 | 3M – 10M | ep_len 持续下降 | 0.6 – 0.9 |
| Phase 4: 收敛 | 10M – 20M | 稳定最优策略 | 0.9 – 0.97 |

`ep_rew_mean` 在 DoorKey 中上限约为 `1 - 0.9*(steps/max_steps)`,接近 0.97 即视为完美。

---

## 7. 执行步骤

```bash
# 1. 装依赖
pip install -r requirements.txt

# 2. 后台启动训练 (用 tmux/nohup,断网不掉)
tmux new -s ppo
python train.py
# Ctrl+B D 脱离

# 3. 实时监控
tensorboard --logdir=logs/ --port=6006

# 4. 训完出图
python plot.py

# 5. 渲染 demo (可选)
python eval.py
```

---

## 8. 报告大纲 (5 页)

| 页 | 内容 |
|---|---|
| 1 | 引言 + 环境介绍 (DoorKey-8x8) + PPO 简述 |
| 2 | 实现细节 + 超参表 + 架构图 (复用第 2 节) |
| 3 | **训练曲线主图** + 5 个阶段的标注与解读 |
| 4 | 辅助指标 (ep_len, entropy, KL) 与行为关联 |
| 5 | What worked / didn't work + 改进方向 + 结论 |

---

## 9. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 训练中途断电/掉网 | `CheckpointCallback` + tmux,可从 `*.zip` 续训 |
| 24h GPU 占用被踢 | 每 500k steps 存档,即使中断也有曲线可用 |
| 策略提前收敛(没素材写) | 换更难环境 `MultiRoom-N4-S5-v0` 续跑 |
| 熵塌缩 / KL 爆炸 | TB 实时盯 `approx_kl`,超 0.05 提前停训分析即可(也是报告素材) |
| MiniGrid 版本 API 差异 | 锁版本:`minigrid==2.3.1`, `stable-baselines3==2.3.2` |

---

## 10. 验收清单 (Definition of Done)

- [ ] `train.py` 跑通 ≥ 20M env_steps
- [ ] `logs/` 含完整 TensorBoard 数据
- [ ] `figures/training_curve.png` 生成成功
- [ ] 至少能在曲线上识别 3 个明显阶段
- [ ] 5 页报告完成,含主图 + 行为分析 + worked/didn't work
