# nasim-knowledge-transfer

Four knowledge transfer methods for reinforcement learning penetration testing agents
in NASim.

## Installation

Python 3.8.10.

```
pip install -r requirements.txt
```

## Contents

Scenario generator, the wrapper classes for reward shaping, action masking and teacher
guidance, the policy distillation setup, the training and evaluation scripts, the
configuration files, and the per-configuration evaluation results.

Training logs and model checkpoints are not included for size reasons.

## Scenario sets

| | |
|---|---|
| `FULL` | all subnets and hosts reachable and visible from the start of the episode |
| `PART` | subnets containing a single host are hidden; two thirds of the generated scenarios are affected |

Host attributes are hidden in both sets and acquired through scans. Seeds 0–99 of each
set are reserved as test instances and excluded from training by a fingerprint of the
initial observation.

## Abbreviations

| | |
|---|---|
| `rs` | reward shaping |
| `am` | action masking |
| `tg` | teacher guidance |
| `ppd` | policy distillation |

**Reward shaping variants** — `base` unmodified, `naive` terminal reward only,
`impulse` bonus for prepared exploits, `act` penalties for redundant actions,
`hier` kill chain progression.

**Action masks** — `poss` currently executable actions only, `nodupe` excludes actions
that already succeeded, `adv` both constraints combined.

**Teacher guidance variants** — `start` teacher acts for the first 5 steps, `end`
teacher acts from step 10 onward, `exp` rule-driven exploit advice, `nscan` rule-driven
subnet scan advice.

## Configuration naming

Runs are named by their hyperparameters, for example `L0.001_N1024_E0.05_C0.2` for a
learning rate of 0.001, 1024 rollout steps, an entropy coefficient of 0.05 and a
clipping range of 0.2.

The nine baseline configurations vary the learning rate (0.01, 0.001, 0.0001) and the
entropy coefficient (0.1, 0.05, 0.001). Rollout steps and clipping range are fixed at
2048 and 0.2.

## Metric

All results are goal rates: the share of the 100 held-out test instances in which the
target host is compromised to root access, measured under deterministic action
selection at the end of the training budget.