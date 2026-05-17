#set page(
  paper: "a4",
  margin: (x: 2.0cm, y: 1.9cm),
  numbering: "1 / 1",
)
#set text(font: "DejaVu Sans", size: 10pt, lang: "en")
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.")
#show heading.where(level: 1): it => block(below: 0.4em, above: 0.55em)[
  #text(size: 12.5pt, weight: "bold")[#it]
]
#show heading.where(level: 2): it => block(below: 0.3em, above: 0.45em)[
  #text(size: 10.5pt, weight: "bold")[#it]
]
#show raw.where(block: true): it => block(
  fill: rgb("#f4f4f4"), inset: 5pt, radius: 3pt, width: 100%,
)[#it]

#align(center)[
  #text(size: 15pt, weight: "bold")[
    The Difficulty Spectrum of MiniGrid: Four PPO Experiments
  ]
  #v(0.2em)
  #text(size: 10pt)[Homework 3 & 4 — Reinforcement Learning]
  #v(0.15em)
  #text(size: 9pt, fill: gray)[Author: Xiangyu Wu · 2026-05-17]
]
#v(0.3em)

= Introduction

This report studies how *task difficulty* affects PPO's ability to learn
on MiniGrid. Rather than committing the whole training budget to one
environment, we ran the same PPO+MLP recipe on four MiniGrid layouts
that differ in grid size, room count, and step budget, and we compare
the resulting learning dynamics. The motivation for this design: our
initial 4 M-step DoorKey-8x8 run converged in $approx$ 14 minutes (far
faster than the PLAN expected), and a subsequent 20 M-step
MultiRoom-N4-S5 run produced zero reward across all 19 532 logged
iterations. Neither outcome on its own gives the multi-phase learning
curve the assignment is built around; together with two intermediate
points, they give a clean picture of where PPO's "sweet spot" lies on
this benchmark family.

*Proximal Policy Optimization* (PPO, Schulman et al., 2017) is an
on-policy gradient method that stabilises the likelihood-ratio policy
gradient by clipping the importance ratio
$r_t(theta) = pi_theta(a_t|s_t) / pi_(theta_(o l d))(a_t|s_t)$ to
$[1-epsilon, 1+epsilon]$. The clipped surrogate
$L^("CLIP") = bb(E)_t[min(r_t hat(A)_t,
"clip"(r_t, 1-epsilon, 1+epsilon) hat(A)_t)]$
is jointly optimised with a value-function loss and an entropy bonus
that keeps exploration alive.

= Method and Hyperparameters

All four runs share an identical pipeline implemented in `train.py`:

#figure(
  image("figures/architecture.png", width: 92%),
  caption: [System architecture (identical across the four
  experiments). Eight parallel `SubprocVecEnv` workers feed a single
  PPO `MlpPolicy` on an L20 GPU; checkpoints are saved every 500 k env
  steps and a held-out `EvalCallback` runs 10 deterministic episodes
  every 100 k env steps.],
) <fig:arch>

The only thing we change across runs is the `ENV_ID` string. Every
other knob is held fixed at the SB3 reference recipe (Table
@tab:hp); the observation is the 7$times$7$times$3 partial view
flattened by `FlatObsWrapper` and the policy is the default
2$times$64 tanh MLP. *No hyperparameter tuning was performed* —
holding hyperparameters constant is the point of this experiment, since
we want to attribute the outcome difference to the *environment*.

#figure(
  table(
    columns: 4, align: (left, right, left, right),
    inset: 4pt, stroke: 0.5pt,
    [*Hyperparameter*], [*Value*], [*Hyperparameter*], [*Value*],
    [Policy], [`MlpPolicy` 2$times$64 tanh], [Discount $gamma$], [0.99],
    [Optimizer], [Adam], [GAE $lambda$], [0.95],
    [Learning rate], [2.5e-4], [Clip range $epsilon$], [0.2],
    [Workers `n_envs`], [8 (SubprocVecEnv)], [Entropy coef], [0.01],
    [Rollout `n_steps`], [128 per worker], [Value coef], [0.5],
    [Total batch], [1 024 transitions], [Max grad norm], [0.5],
    [Minibatch size], [256], [Device], [`cuda` (NVIDIA L20)],
    [Update epochs], [4], [SB3 version], [2.3.2],
  ),
  caption: [PPO hyperparameters, *identical across all four
  experiments*. These match the SB3 reference recipe for MiniGrid and
  were not tuned in this work.],
) <tab:hp>

#figure(
  table(
    columns: 6, align: (left, left, right, right, right, right),
    inset: 4pt, stroke: 0.5pt,
    [*Environment*], [*Layout / max_steps*], [*env_steps*], [*Wall-clock*], [*FPS*], [*Outcome*],
    [DoorKey-8x8], [8$times$8 grid, 640 steps], [4.07 M], [15 min], [4 664], [#text(fill: rgb("#117a3a"))[converged 0.975]],
    [MultiRoom-N2-S4], [2 rooms $times$ size 4, 40 steps], [17.68 M], [65 min], [4 530], [#text(fill: rgb("#117a3a"))[converged 0.843]],
    [DoorKey-16x16], [16$times$16 grid, 2560 steps], [20.00 M], [99 min], [3 369], [#text(fill: rgb("#a66800"))[partial 0.014]],
    [MultiRoom-N4-S5], [4 rooms $times$ size 5, 120 steps], [20.00 M], [71 min], [4 741], [#text(fill: rgb("#a01a1a"))[failed 0.000]],
  ),
  caption: [The four environments, sorted by *empirical* difficulty for
  PPO+MLP. "Converged" / "partial" / "failed" labels are read from
  `eval/mean_reward` at the end of training.],
) <tab:envs>

== Reward structure
All four environments share the same sparse-reward template: the agent
receives $r = 1 - 0.9 dot ("steps" / "max_steps")$ on reaching the
goal, and zero otherwise. The reward ceiling therefore depends on how
quickly the geometric shortest path can be executed; for DoorKey-8x8
that ceiling is $approx$ 0.97 and for MultiRoom-N2-S4 it is
$approx$ 0.86 (matching the observed plateau).

= Results

#figure(
  image("figures/cmp_reward.png", width: 100%),
  caption: [`rollout/ep_rew_mean` for all four runs, smoothed with a
  51-point moving average. The x-axis is `env_steps`. Both DoorKey-8x8
  and MultiRoom-N2-S4 converge to their reward ceilings in under 500 k
  steps; DoorKey-16x16 shows a barely-perceptible 1–2 % learning
  signal; MultiRoom-N4-S5 stays flat at zero the entire 20 M-step
  run.],
) <fig:cmp_rew>

The four runs cleanly stratify into three regimes (Fig. @fig:cmp_rew):

1. *Solved* (DoorKey-8x8, MultiRoom-N2-S4). Both reach their
   geometric reward ceiling. DoorKey-8x8's smoothed reward crosses 0.1
   at $approx$ 156 k env steps, 0.5 at 276 k, 0.9 at 329 k, and 0.95 at
   376 k. The transition from random to near-optimal is therefore
   compressed into a $approx 2 dot 10^5$-step window. MultiRoom-N2-S4
   converges even faster.
2. *Partially learning* (DoorKey-16x16). The first non-zero
   `ep_rew_mean` appears at step 317 k. Across the full 20 M steps,
   24.3 % of logged iterations have non-zero mean reward, with peak
   smoothed reward 0.038. The *deterministic* policy never solves the
   task — all 200 evaluation calls return exactly 0.000 — so the
   sparse successes are essentially residual stochastic-policy luck,
   not a committed solution.
3. *Failed* (MultiRoom-N4-S5). Zero reward in 19 532 logged
   iterations and all 200 eval calls. The agent did not see a single
   successful episode in 20 M env_steps.

#figure(
  table(
    columns: 5, align: (left, right, right, right, right),
    inset: 4pt, stroke: 0.5pt,
    [*Phase (DoorKey-8x8)*], [*env_steps*], [*`ep_rew`*], [*`ep_len`*], [*entropy*],
    [P0 — exploration],     [0 – 100 k],   [< 0.04],     [$approx$ 620], [$-1.85$],
    [P1 — causal learning], [100 k – 300 k], [0.04 $arrow$ 0.79], [620 $arrow$ 145], [$-1.45$],
    [P2 — path optimisation], [300 k – 500 k], [0.79 $arrow$ 0.97], [145 $arrow$ 21], [$-0.41$],
    [P3 — convergence],     [500 k – 4 M], [$approx$ 0.975], [17 – 19], [$-0.02$],
  ),
  caption: [Four-phase decomposition of the DoorKey-8x8 learning
  curve, read from the smoothed `ep_rew_mean` crossings. The phases
  are visually compressed in Fig. @fig:cmp_rew because the entire
  story finishes in the leftmost $approx$ 2.5 % of the x-axis.],
)

== Auxiliary signals

#grid(
  columns: (1fr, 1fr), column-gutter: 0.5em, row-gutter: 0.4em,
  [
    #figure(
      image("figures/cmp_length.png", width: 100%),
      caption: [Episode length. The collapse to single-digit values on
      the solved tasks lags the reward rise by $approx$ 50 k steps:
      the agent first learns to *solve*, then to solve *quickly*.],
    )
  ],
  [
    #figure(
      image("figures/cmp_entropy.png", width: 100%),
      caption: [Policy entropy loss. Solved tasks collapse from
      $approx -1.95$ to $approx -0.02$ (near-deterministic policy).
      The failing tasks retain high entropy because the entropy bonus
      is never overwhelmed by an advantage signal.],
    )
  ],
)

The entropy diagnostic is particularly revealing. On the two
*failed/partial* tasks, the policy keeps entropy high
(MultiRoom-N4-S5 ends at $-1.61$, DoorKey-16x16 at $-1.20$) — not
because the algorithm chose to keep exploring, but because there is no
positive advantage to push the action distribution toward. The
*explained variance* of the value function tells the same story: on
DoorKey-16x16 it spent the whole run hovering around 0 (final 0.29),
because there are too few non-zero returns to fit.

#grid(
  columns: (1fr, 1fr), column-gutter: 0.5em,
  [
    #figure(
      image("figures/cmp_final_bar.png", width: 100%),
      caption: [Final (last-50-iteration mean) vs. peak training
      reward per environment. The gap between the two on
      DoorKey-16x16 (peak $approx$ 0.038, final $approx$ 0.014) is
      itself evidence that the agent *can* sometimes hit the goal —
      but cannot make the success stick.],
    ) <fig:bar>
  ],
  [
    #figure(
      image("figures/cmp_eval.png", width: 100%),
      caption: [Held-out deterministic `eval/mean_reward`. Confirms
      that DoorKey-16x16's training-reward signal does not transfer
      to a deterministic policy — the eval line is identically zero
      throughout, in contrast to DoorKey-8x8's clean climb.],
    ) <fig:eval>
  ],
)

== A successful rollout

To make the "solved" regime concrete, three deterministic rollouts of
the best DoorKey-8x8 policy yielded returns of 0.976 / 0.978 / 0.970
with lengths 17 / 16 / 21 steps:

#figure(
  image("figures/rollout_ep0.png", width: 88%),
  caption: [Frames from a deterministic DoorKey-8x8 rollout: the
  policy turns to the key, picks it up, walks to the door, unlocks
  it, and reaches the goal in 17 environment steps.],
)

= Discussion: When PPO works on MiniGrid, and when it doesn't

*What worked.*
- *Off-the-shelf SB3 hyperparameters* immediately solved DoorKey-8x8
  and MultiRoom-N2-S4 — no tuning needed.
- *`FlatObsWrapper` + MLP* over CNN: the symbolic 7$times$7$times$3
  observation has no spatial structure that a 2$times$64 MLP cannot
  digest. Switching to a CNN would slow training without changing the
  asymptote on the solved tasks.
- *`SubprocVecEnv(n=8)`* gave the throughput needed to make 20 M-step
  runs cheap (60–100 min wall-clock each).
- *Held-out `EvalCallback`* immediately exposed the
  stochastic-vs-deterministic gap on DoorKey-16x16 — without it we
  might have mistaken the 1.6 % training reward for actual learning.

*What did not work, and why.*
- *DoorKey-16x16.* The geometric reward ceiling at this size requires
  $approx$ 30 step trajectories from a 2560-step budget; random
  policies hit the goal with probability $approx 10^(-6)$ per
  trajectory. Even with 8 envs $times$ 20 M steps that is barely
  enough to seed a few successes, and the resulting advantage signal
  is too noisy to lift the policy out of the random regime. The
  policy *did* discover the goal sporadically (peak 3.8 % training
  reward) but never committed to a deterministic solution.
- *MultiRoom-N4-S5.* Worse: the 120-step budget plus four random
  rooms plus narrow doors makes the random success probability
  effectively zero. The agent never saw a non-zero return, so the
  entire 20 M-step training was a no-op. This is the canonical
  *deep-exploration* failure mode that motivated intrinsic-curiosity
  methods like ICM and RND.

*Surprises.* The original PLAN expected a graceful 4-phase learning
curve across 20 M steps. In practice MiniGrid presents PPO with a
*binary* difficulty cliff: either the random policy can stumble onto a
success in $approx 10^5$ trajectories (and PPO then sharpens it
within $approx 10^5$ env steps), or it cannot and PPO produces no
learning at all. There is essentially no "long, gradual" regime in
this benchmark family under fixed hyperparameters.

*Improvement directions.*
1. *Anneal `ent_coef`* (e.g. 0.05 $arrow$ 0.001) to push more
   exploration early — likely the cheapest fix for DoorKey-16x16.
2. *Switch to `RecurrentPPO`* (sb3-contrib): an LSTM lets the agent
   integrate evidence across the partial-observation window, which
   matters more as the grid grows.
3. *Add an intrinsic-curiosity bonus* (ICM / RND) for the
   MultiRoom-N4-S5 deep-exploration setting.
4. *Curriculum* from N2-S4 $arrow$ N3-S4 $arrow$ N4-S5 would let the
   policy bootstrap from the easy task — a single hard environment is
   the wrong abstraction.

*Conclusion.* PPO with SB3 defaults occupies a narrow operating point
on MiniGrid: it solves DoorKey-8x8 and MultiRoom-N2-S4 in
$<$ 500 k env steps and to within $approx$ 2 % of the geometric
ceiling, but at the next difficulty step it falls off a cliff into
near-zero performance. The four-experiment view shows that the PLAN's
"24-hour single-task benchmark" framing is mis-specified for this
environment family — a *curriculum or exploration-shaping* axis would
be a strictly more informative thing to study with the same compute
budget.
