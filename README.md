# Reinforcement Learning: Endogenous Exploration with Intrinsic Curiosity

Code accompanying the paper **[Endogenous Exploration in Reinforcement Learning with Intrinsic Curiosity](https://arxiv.org/abs/2609.05650)** by Armando Vieira (University of Tartu), arXiv:2609.05650 [cs.LG], 2026. ([PDF](https://arxiv.org/pdf/2609.05650))

## Summary

Most RL methods treat exploration as something bolted on from outside: ε-greedy noise, entropy bonuses, or prediction-error curiosity (ICM, RND). This work proposes the **Experiential Coherence Framework (ECF)**, where exploration emerges *endogenously* from the agent's own internal dynamics. The agent regulates coherence between *reach* (possible continuations), *yield* (current situational demands) and a sedimented *memory* trace. The resulting incoherence and novelty signals produce an intrinsic reward:

$$r^{tot}_t = r^{ext}_t + \lambda_t \, \psi(I_t) \, N_t$$

The central hypothesis, the **Curiosity Window conjecture**, is that effective exploration happens at *intermediate* levels of incoherence: too little and the agent is trapped, too much and it fragments. The framework targets non-stationary environments with sparse, delayed or absent rewards.

## Key findings

- **Curiosity Window confirmed.** The C2 metric (coherent basins × global overlap × transition rate) peaks at intermediate noise, with a drop-off of more than 1.5× on both sides.
- **Distinct from Active Inference.** Active Inference agents show no analogous peak under the same analysis.
- **LunarLander-v2 (discrete).** ECF-RichMem was the strongest agent in all four phases versus ECF and ICM (10 seeds):

  | Agent | P1 Train | P2 Retain | P3 Wind | P4 Recover |
  |---|---|---|---|---|
  | ECF-RichMem | **16.3 ± 13.4** | **40.0 ± 10.3** | **−16.7 ± 12.7** | **5.7 ± 16.2** |
  | ECF | −6.4 ± 7.5 | −34.2 ± 10.1 | −58.3 ± 20.7 | −45.3 ± 35.2 |
  | ICM | −18.3 ± 5.3 | −18.6 ± 35.7 | −90.3 ± 32.4 | −72.3 ± 27.9 |

- **BipedalWalker-v3 (continuous).** PPO and PPO+ECF perform similarly under stable conditions, but under a gravity disturbance PPO collapses while PPO+ECF stays strongly positive:

  | Agent | P1 Train | P2 Retain | P3 ΔGravity | P4 Recover |
  |---|---|---|---|---|
  | PPO | **31.1 ± 45.9** | **42.6 ± 60.3** | −20.3 ± 36.6 | 3.1 ± 58.2 |
  | PPO+ECF | 29.0 ± 75.9 | 37.4 ± 80.2 | **45.3 ± 80.3** | **5.4 ± 85.3** |

- **Cross-experiment pattern.** The ECF advantage is smallest in stable, well-rewarded settings and largest when the environment shifts or reward is absent, and it grows going from discrete to continuous action spaces.
- **Limitations.** PPO+ECF shows much higher variance than PPO, basin dwell times are short, and only the C2 metric clearly supports the conjecture.

## Code

`ppo_ecf_bipedalwalker.py` runs the PPO vs PPO+ECF experiment on BipedalWalker-v3 using a phased protocol (nominal training, retention, gravity disturbance, recovery). The hybrid ECF module combines a continuous latent projection, solidification / visit density, asymmetric sedimentation and erosion, multi-timescale memory with a curriculum, an adaptive λ gate, an action smoothness penalty, basin coherence accumulation, and an intrinsic bonus of incoherence × novelty × coherence. It reports Welch's t-test, Cohen's d and a bootstrap 95% CI, and saves plots to disk.

### Install

```bash
conda install -c conda-forge swig -y
pip install "gymnasium[box2d]" stable-baselines3 scipy matplotlib torch
```

### Run

```bash
python ppo_ecf_bipedalwalker.py --help
```

## Citation

```bibtex
@article{vieira2026endogenous,
  title   = {Endogenous Exploration in Reinforcement Learning with Intrinsic Curiosity},
  author  = {Vieira, Armando},
  journal = {arXiv preprint arXiv:2609.05650},
  year    = {2026}
}
```
