"""
PPO vs PPO+ECF on BipedalWalker-v3
===================================
3-phase experiment:
  Phase 1 — stable environment (normal gravity)
  Phase 2 — gravity shift (reward active)
  Phase 3 — gravity shift + reward zeroed (pure exploration test)

New ECF combines:
  - Continuous latent projection (Code 1: no aliasing)
  - Solidification / visit density (Code 2: memory consolidation)
  - Asymmetric sedimentation / erosion (Code 2)
  - Multi-timescale memory blend with curriculum (Code 1)
  - Adaptive lambda gate (Code 1)
  - Action smoothness penalty (Code 1)
  - Basin coherence accumulation (Code 2)
  - Intrinsic bonus: incoherence × novelty × coherence (hybrid)

Statistical output:
  - Welch's t-test + Cohen's d
  - Bootstrap 95% CI on Phase 3 difference
  - Full plots saved to disk

Install:
    conda install -c conda-forge swig -y
    pip install "gymnasium[box2d]" stable-baselines3 scipy matplotlib torch
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# ─────────────────────────── Hyperparameters ───────────────────────────

DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_SEEDS             = 50
PHASE1_EPISODES     = 400   # P1: standard training, nominal gravity
PHASE2_EPISODES     = 150   # P2: retention, nominal gravity
PHASE3_EPISODES     = 200   # P3: disturbance - shifted gravity
PHASE4_EPISODES     = 200   # P4: recovery - nominal gravity restored

N_ENVS              = 4     # parallel envs per agent (matches original)
STEPS_PER_UPDATE    = 2048  # steps per env per update (restored from original)
PPO_EPOCHS          = 10
MINI_BATCH_SIZE     = 64
GAMMA               = 0.99
GAE_LAMBDA          = 0.95
CLIP_EPS            = 0.2
LR_ACTOR            = 3e-4
LR_CRITIC           = 1e-3  # faster value learning (separate from policy)
ENTROPY_COEF        = 0.01
VF_COEF             = 0.5
MAX_GRAD_NORM       = 0.5

NOMINAL_GRAVITY     = -10.0  # BipedalWalker default
DISTURB_GRAVITY     = -15.0  # Phase 3 disturbance (50% stronger)


# ─────────────────────────── ECF Config ────────────────────────────────

@dataclass
class ECFConfig:
    # Latent space
    latent_dim: int        = 24
    proj_std: float        = 0.5

    # Experience EMA
    alpha_y: float         = 0.15

    # Multi-timescale memory
    alpha_fast: float      = 0.10
    alpha_med: float       = 0.02
    alpha_slow: float      = 0.002
    blend_pivot: int       = 10_000   # step at which blend shifts fast→slow

    # Expectation update
    pi_lr: float           = 0.20

    # Solidification (visit density)
    visit_decay: float     = 0.995
    visit_alpha: float     = 0.005
    solidification_k: float = 5.0    # steepness of 1/(1+k*density) curve
    solid_min: float       = 0.10
    solid_max: float       = 1.00

    # Asymmetric sedimentation / erosion
    sediment_rate: float   = 0.03
    erosion_rate: float    = 0.01

    # Coherence accumulation
    coh_decay: float       = 0.995
    coh_alpha: float       = 0.005

    # Adaptive lambda gate
    lam_base: float        = 0.00
    lam_max: float         = 0.03
    lam_incoherence_scale: float = 2.0

    # Intrinsic bonus composition
    novelty_scale: float   = 0.50
    coherence_scale: float = 0.25
    trend_scale: float     = 0.10

    # Clipping & penalties
    intrinsic_clip: float  = 2.0
    action_penalty_scale: float = 0.01

    seed: int              = 0


# ─────────────────────────── Actor-Critic ──────────────────────────────

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
        )
        self.mu_head  = nn.Linear(hidden, act_dim)
        self.log_std  = nn.Parameter(torch.zeros(act_dim))
        self.v_head   = nn.Linear(hidden, 1)

        # orthogonal init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)

    def forward(self, x: torch.Tensor):
        h   = self.shared(x)
        mu  = torch.tanh(self.mu_head(h))
        std = self.log_std.exp().expand_as(mu)
        return mu, std, self.v_head(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        mu, std, v = self.forward(obs)
        dist = Normal(mu, std)
        a    = dist.sample()
        logp = dist.log_prob(a).sum(-1)
        return a.cpu().numpy(), logp.cpu().numpy(), v.cpu().numpy()

    def evaluate(self, obs: torch.Tensor, acts: torch.Tensor):
        mu, std, v = self.forward(obs)
        dist    = Normal(mu, std)
        logp    = dist.log_prob(acts).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logp, v, entropy


# ─────────────────────────── Hybrid ECF Module ─────────────────────────

class HybridECF:
    """
    Continuous-latent ECF (Code-1 geometry) enriched with:
      - visit-density solidification        (Code-2)
      - asymmetric sedimentation/erosion    (Code-2)
      - coherence accumulation              (Code-2)
      - adaptive lambda gate                (Code-1)
      - multi-timescale blend curriculum    (Code-1)
      - action smoothness penalty           (Code-1)
    """

    def __init__(self, obs_dim: int, act_dim: int, cfg: ECFConfig):
        self.cfg     = cfg
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        d            = cfg.latent_dim
        rng          = np.random.default_rng(cfg.seed)

        # fixed random projection obs → latent
        self.proj = rng.normal(0.0, cfg.proj_std, (d, obs_dim)).astype(np.float32)

        self._reset_state()

    # ── state ──

    def _reset_state(self):
        d = self.cfg.latent_dim
        self.pi       = np.zeros(d, np.float32)
        self.y        = np.zeros(d, np.float32)
        self.m_fast   = np.zeros(d, np.float32)
        self.m_med    = np.zeros(d, np.float32)
        self.m_slow   = np.zeros(d, np.float32)
        self.m_eff    = np.zeros(d, np.float32)   # sedimented effective memory
        self.visit_density  = np.zeros(d, np.float32)
        self.coh_accum      = np.zeros(d, np.float32)
        self.prev_p         = 0.0
        self.prev_action    = np.zeros(self.act_dim, np.float32)
        self.steps          = 0

        # diagnostics
        self.I_history     = deque(maxlen=500)
        self.G_history     = deque(maxlen=500)
        self.bonus_history = deque(maxlen=500)
        self.sigma_history = deque(maxlen=500)

    def reset(self):
        self._reset_state()

    # ── helpers ──

    def _latent(self, obs: np.ndarray) -> np.ndarray:
        return np.tanh(self.proj @ np.asarray(obs, np.float32).reshape(-1))

    def _solidification(self) -> np.ndarray:
        cfg   = self.cfg
        solid = cfg.solid_min + (cfg.solid_max - cfg.solid_min) / (
            1.0 + cfg.solidification_k * self.visit_density)
        return np.clip(solid, cfg.solid_min, cfg.solid_max)

    def _effective_memory(self) -> np.ndarray:
        t       = min(1.0, self.steps / self.cfg.blend_pivot)
        m_early = 0.6 * self.m_fast + 0.3 * self.m_med + 0.1 * self.m_slow
        m_late  = 0.2 * self.m_fast + 0.3 * self.m_med + 0.5 * self.m_slow
        return (1.0 - t) * m_early + t * m_late

    def _incoherence(self) -> float:
        return float(
            np.mean((self.pi - self.y) ** 2) +
            np.mean((self.y  - self.m_eff) ** 2)
        )

    def _coherence(self) -> float:
        return float(np.exp(-50*np.mean((self.pi - self.m_eff) ** 2)))

    def _novelty(self) -> float:
        return float(np.mean(np.abs(self.y - self.m_eff)))

    # ── per-step update called from rollout loop ──

    def step_update(self, obs: np.ndarray, action: np.ndarray) -> float:
        """
        Update all ECF state for one env step.
        Returns the scalar intrinsic reward for this step.
        """
        cfg  = self.cfg
        z    = self._latent(obs)
        self.steps += 1

        # ── experience EMA ──
        self.y = (1.0 - cfg.alpha_y) * self.y + cfg.alpha_y * z

        # ── multi-timescale memory ──
        self.m_fast += cfg.alpha_fast  * (self.y      - self.m_fast)
        self.m_med  += cfg.alpha_med   * (self.m_fast  - self.m_med)
        self.m_slow += cfg.alpha_slow  * (self.m_med   - self.m_slow)
        m_blend      = self._effective_memory()

        # ── visit density & solidification ──
        act_energy  = float(np.mean(action ** 2))
        activity    = np.abs(self.y) * np.clip(act_energy, 0, 1)
        self.visit_density = (cfg.visit_decay * self.visit_density +
                               cfg.visit_alpha * activity)

        solid = self._solidification()

        # ── asymmetric sedimentation / erosion ──
        delta   = self.y - self.m_eff
        aligned = (self.y * self.m_eff) > 0
        alpha_s = np.where(aligned,
                           cfg.sediment_rate * solid,
                           cfg.erosion_rate  * solid)
        self.m_eff += alpha_s * delta

        # ── blend sedimented memory with multi-timescale ──
        self.m_eff = 0.6 * self.m_eff + 0.4 * m_blend

        # ── coherence accumulation → extra visit stabilisation ──
        local_coh = np.exp(
            -((self.pi - self.y) ** 2 + (self.y - self.m_eff) ** 2))
        self.coh_accum = (cfg.coh_decay * self.coh_accum +
                          cfg.coh_alpha * local_coh)
        self.visit_density += 0.001 * self.coh_accum
        self.visit_density  = np.clip(self.visit_density, 0.0, 10.0)

        # ── expectation update ──
        self.pi += cfg.pi_lr * (self.y - self.pi)

        # ── intrinsic signals ──
        I         = self._incoherence()
        G         = self._coherence()
        novelty   = self._novelty()

        p_ecf  = cfg.novelty_scale * novelty + cfg.coherence_scale * G
        trend  = p_ecf - self.prev_p
        self.prev_p = p_ecf

        # ── adaptive lambda ──
        lam = cfg.lam_base + (cfg.lam_max - cfg.lam_base) * (
            1.0 - np.exp(-cfg.lam_incoherence_scale * I))

        # ── action smoothness penalty ──
        act_penalty = cfg.action_penalty_scale * float(
            np.mean((action - self.prev_action) ** 2))
        self.prev_action = np.asarray(action, np.float32).copy()

        reward_int = lam * (p_ecf + cfg.trend_scale * trend) - act_penalty
        reward_int = float(np.clip(reward_int,
                                   -cfg.intrinsic_clip, cfg.intrinsic_clip))

        # diagnostics
        self.I_history.append(I)
        self.G_history.append(G)
        self.bonus_history.append(reward_int)

        return reward_int

    # ── batch update after PPO rollout (optional field smoothing) ──

    def batch_update(self, obs_batch: np.ndarray, act_batch: np.ndarray):
        """Light-weight pass to keep field statistics in sync post-update."""
        for obs, act in zip(obs_batch, act_batch):
            self.step_update(obs, act)


# ─────────────────────────── PPO Update ────────────────────────────────

def ppo_update(ac: ActorCritic,
               optimizer: optim.Optimizer,
               rollout: dict) -> dict:
    obs_t  = torch.tensor(rollout["obs"],  dtype=torch.float32, device=DEVICE)
    act_t  = torch.tensor(rollout["act"],  dtype=torch.float32, device=DEVICE)
    ret_t  = torch.tensor(rollout["ret"],  dtype=torch.float32, device=DEVICE)
    adv_t  = torch.tensor(rollout["adv"],  dtype=torch.float32, device=DEVICE)
    logp_t = torch.tensor(rollout["logp"], dtype=torch.float32, device=DEVICE)

    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    n     = obs_t.shape[0]
    metrics = {"loss_pi": [], "loss_v": [], "entropy": []}

    for _ in range(PPO_EPOCHS):
        idx = np.random.permutation(n)
        for start in range(0, n, MINI_BATCH_SIZE):
            mb      = idx[start:start + MINI_BATCH_SIZE]
            new_logp, new_v, ent = ac.evaluate(obs_t[mb], act_t[mb])
            ratio   = (new_logp - logp_t[mb]).exp()
            surr1   = ratio * adv_t[mb]
            surr2   = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * adv_t[mb]
            loss_pi = -torch.min(surr1, surr2).mean()
            loss_v  = (new_v - ret_t[mb]).pow(2).mean()
            loss    = loss_pi + VF_COEF * loss_v - ENTROPY_COEF * ent.mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            metrics["loss_pi"].append(loss_pi.item())
            metrics["loss_v"].append(loss_v.item())
            metrics["entropy"].append(ent.mean().item())

    return {k: float(np.mean(v)) for k, v in metrics.items()}


# ─────────────────────────── Rollout Collection ────────────────────────

def collect_rollout(
    env: gym.Env,
    ac: ActorCritic,
    ecf: Optional[HybridECF],
    steps: int = STEPS_PER_UPDATE,
    zero_reward: bool = False,
) -> Tuple[dict, List[float]]:
    obs_buf, act_buf, rew_buf, val_buf, logp_buf, done_buf = [], [], [], [], [], []
    ep_rewards: List[float] = []
    ep_ret = 0.0

    obs, _ = env.reset()

    for _ in range(steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        act_np, logp_np, val_np = ac.act(obs_t)
        act = act_np[0]
        clipped = np.clip(act, env.action_space.low, env.action_space.high)

        next_obs, rew, terminated, truncated, _ = env.step(clipped)
        done = terminated or truncated

        ext_rew = 0.0 if zero_reward else float(rew)
        int_rew = ecf.step_update(obs, act) if ecf is not None else 0.0
        total   = ext_rew + int_rew

        obs_buf.append(obs.copy())
        act_buf.append(act.copy())
        rew_buf.append(total)
        val_buf.append(val_np[0])
        logp_buf.append(logp_np[0])
        done_buf.append(float(done))

        ep_ret += float(rew)   # always track true reward
        if done:
            ep_rewards.append(ep_ret)
            ep_ret = 0.0
            obs, _ = env.reset()
        else:
            obs = next_obs

    # GAE
    obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    _, _, last_val = ac.act(obs_t)
    vals  = np.append(np.array(val_buf), last_val[0])
    rews  = np.array(rew_buf)
    dones = np.array(done_buf)
    advs  = np.zeros_like(rews)
    gae   = 0.0
    for t in reversed(range(len(rews))):
        delta = rews[t] + GAMMA * vals[t + 1] * (1 - dones[t]) - vals[t]
        gae   = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
        advs[t] = gae
    rets = advs + vals[:-1]

    rollout = {
        "obs":  np.array(obs_buf,  dtype=np.float32),
        "act":  np.array(act_buf,  dtype=np.float32),
        "ret":  rets.astype(np.float32),
        "adv":  advs.astype(np.float32),
        "logp": np.array(logp_buf, dtype=np.float32),
    }
    return rollout, ep_rewards


# ─────────────────────────── Vectorised Rollout ────────────────────────

def collect_rollout_vec(
    vec_env,
    ac: ActorCritic,
    ecf: Optional[HybridECF],
    steps_per_env: int = STEPS_PER_UPDATE,
    zero_reward: bool = False,
) -> Tuple[dict, List[float]]:
    """Collect rollouts from N_ENVS parallel envs."""
    n_envs = vec_env.num_envs
    total_steps = steps_per_env * n_envs

    obs_buf, act_buf, rew_buf, val_buf, logp_buf, done_buf = [], [], [], [], [], []
    ep_rewards: List[float] = []
    ep_rets = np.zeros(n_envs, dtype=np.float32)

    obs = vec_env.reset()  # (n_envs, obs_dim)

    for _ in range(steps_per_env):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        act_np, logp_np, val_np = ac.act(obs_t)  # each (n_envs,)

        # clip actions per env
        low  = vec_env.action_space.low
        high = vec_env.action_space.high
        clipped = np.clip(act_np, low, high)

        next_obs, rews, dones, infos = vec_env.step(clipped)

        for i in range(n_envs):
            ext_rew = 0.0 if zero_reward else float(rews[i])
            int_rew = ecf.step_update(obs[i], act_np[i]) if ecf is not None else 0.0

            obs_buf.append(obs[i].copy())
            act_buf.append(act_np[i].copy())
            rew_buf.append(ext_rew + int_rew)
            val_buf.append(val_np[i])
            logp_buf.append(logp_np[i])
            done_buf.append(float(dones[i]))

            ep_rets[i] += float(rews[i])
            if dones[i]:
                # VecEnv auto-resets; retrieve true terminal reward from info
                ep_r = infos[i].get("episode", {}).get("r", ep_rets[i])
                ep_rewards.append(float(ep_r))
                ep_rets[i] = 0.0

        obs = next_obs

    # GAE over the flattened buffer
    obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
    _, _, last_vals = ac.act(obs_t)  # (n_envs,)

    rews  = np.array(rew_buf,  dtype=np.float32)
    vals  = np.array(val_buf,  dtype=np.float32)
    dones = np.array(done_buf, dtype=np.float32)

    # bootstrap last value per env
    last_val_flat = np.repeat(last_vals, 1)  # keep shape simple
    # rebuild per-env sequences for correct GAE
    advs = np.zeros(len(rews), dtype=np.float32)
    for env_i in range(n_envs):
        idx   = np.arange(env_i, len(rews), n_envs)  # this env's indices
        r_i   = rews[idx]
        v_i   = np.append(vals[idx], last_vals[env_i])
        d_i   = dones[idx]
        gae   = 0.0
        for t in reversed(range(len(r_i))):
            delta  = r_i[t] + GAMMA * v_i[t + 1] * (1 - d_i[t]) - v_i[t]
            gae    = delta + GAMMA * GAE_LAMBDA * (1 - d_i[t]) * gae
            advs[idx[t]] = gae

    rets = advs + vals

    rollout = {
        "obs":  np.array(obs_buf,  dtype=np.float32),
        "act":  np.array(act_buf,  dtype=np.float32),
        "ret":  rets,
        "adv":  advs,
        "logp": np.array(logp_buf, dtype=np.float32),
    }
    return rollout, ep_rewards


# ─────────────────────────── Run One Seed ──────────────────────────────

def run_seed(seed: int, use_ecf: bool, ecf_cfg: ECFConfig) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)

    env_tmp = gym.make("BipedalWalker-v3")
    obs_dim = env_tmp.observation_space.shape[0]
    act_dim = env_tmp.action_space.shape[0]
    env_tmp.close()

    ac        = ActorCritic(obs_dim, act_dim).to(DEVICE)
    optimizer = optim.Adam([
        {'params': list(ac.shared.parameters()) +
                   list(ac.mu_head.parameters()) +
                   [ac.log_std],             'lr': LR_ACTOR},
        {'params': ac.v_head.parameters(),   'lr': LR_CRITIC},
    ])
    ecf       = HybridECF(obs_dim, act_dim, ecf_cfg) if use_ecf else None

    results = {
        "phase1": [], "phase2": [], "phase3": [], "phase4": [],
        "I_history": [], "G_history": [], "bonus_history": [],
    }

    def _make_env(rank: int, disturbed: bool):
        def _init():
            e = gym.make("BipedalWalker-v3")
            e.reset(seed=seed + rank)
            gravity = DISTURB_GRAVITY if disturbed else NOMINAL_GRAVITY
            e.unwrapped.world.gravity = (0, gravity)
            return e
        return _init

    def run_phase(phase_key: str, disturbed: bool, target_eps: int):
        from stable_baselines3.common.vec_env import DummyVecEnv
        vec_env = DummyVecEnv([_make_env(i, disturbed) for i in range(N_ENVS)])

        ep_count = 0
        while ep_count < target_eps:
            rollout, ep_rews = collect_rollout_vec(
                vec_env, ac, ecf, STEPS_PER_UPDATE, zero_reward=False)
            ppo_update(ac, optimizer, rollout)
            results[phase_key].extend(ep_rews)
            ep_count += len(ep_rews)

        vec_env.close()

        if ecf is not None:
            results["I_history"].extend(list(ecf.I_history))
            results["G_history"].extend(list(ecf.G_history))
            results["bonus_history"].extend(list(ecf.bonus_history))

    # P1: standard training, nominal gravity
    run_phase("phase1", disturbed=False, target_eps=PHASE1_EPISODES)
    # P2: retention — nominal gravity, tests stability of acquired policy
    run_phase("phase2", disturbed=False, target_eps=PHASE2_EPISODES)
    # P3: disturbance — gravity shifted to DISTURB_GRAVITY
    run_phase("phase3", disturbed=True,  target_eps=PHASE3_EPISODES)
    # P4: recovery — nominal gravity restored
    run_phase("phase4", disturbed=False, target_eps=PHASE4_EPISODES)

    return results


# ─────────────────────────── Statistics ────────────────────────────────

def last_n_mean(ep_list: List[float], n: int = 50) -> float:
    arr = ep_list[-n:] if len(ep_list) >= n else ep_list
    return float(np.mean(arr)) if arr else float("nan")


def welch_and_cohens_d(a: np.ndarray, b: np.ndarray):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    pooled_std = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    d = (a.mean() - b.mean()) / (pooled_std + 1e-12)
    return float(t), float(p), float(d)


def bootstrap_ci(a: np.ndarray, b: np.ndarray,
                 n_boot: int = 10_000, alpha: float = 0.05):
    rng   = np.random.default_rng(42)
    diffs = []
    for _ in range(n_boot):
        ia = rng.choice(len(a), len(a), replace=True)
        ib = rng.choice(len(b), len(b), replace=True)
        diffs.append(a[ia].mean() - b[ib].mean())
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), diffs


def print_stats(all_results: dict):
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS — Welch's t-test (two-sided) + Cohen's d")
    print("=" * 70)

    phases_info = {
        "phase1": "P1: Standard training",
        "phase2": "P2: Retention (nominal)",
        "phase3": "P3: Disturbance (wind)",
        "phase4": "P4: Recovery (nominal restored)",
    }
    for phase, label in phases_info.items():
        ppo_vals = np.array(all_results["PPO"][phase])
        ecf_vals = np.array(all_results["PPO+ECF"][phase])
        t, p, d = welch_and_cohens_d(ecf_vals, ppo_vals)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"\n{label}")
        print(f"  PPO    : {ppo_vals.mean():8.1f} ± {ppo_vals.std():.1f}")
        print(f"  PPO+ECF: {ecf_vals.mean():8.1f} ± {ecf_vals.std():.1f}")
        print(f"  Welch  : t={t:6.2f}, p={p:.4f}, d={d:.3f}  {sig}")

    print("\n" + "-" * 70)
    print("BOOTSTRAP 95% CI  —  Phase 4 recovery  (PPO+ECF − PPO)")
    ecf4 = np.array(all_results["PPO+ECF"]["phase4"])
    ppo4 = np.array(all_results["PPO"]["phase4"])
    lo, hi, diffs = bootstrap_ci(ecf4, ppo4)
    mean_diff = ecf4.mean() - ppo4.mean()
    sig = "Yes (CI excludes 0)" if (lo > 0 or hi < 0) else "No (CI includes 0)"
    print(f"  Mean diff : {mean_diff:.1f}")
    print(f"  95% CI    : [{lo:.1f}, {hi:.1f}]")
    print(f"  Significant: {sig}")

    print("\n" + "-" * 70)
    print("RECOVERY GAIN  —  P4 - P3  (how much each agent bounced back)")
    for agent in ["PPO", "PPO+ECF"]:
        p3 = np.array(all_results[agent]["phase3"])
        p4 = np.array(all_results[agent]["phase4"])
        delta = p4 - p3
        print(f"  {agent:8s}: {delta.mean():+.1f} ± {delta.std():.1f}")

    return diffs, lo, hi, mean_diff


# ─────────────────────────── Plotting ──────────────────────────────────

def smooth(arr: List[float], w: int = 10) -> np.ndarray:
    a = np.array(arr, dtype=np.float32)
    if len(a) < w:
        return a
    kernel = np.ones(w) / w
    return np.convolve(a, kernel, mode="valid")


def plot_all(all_results: dict,
             raw_per_seed: dict,
             bootstrap_diffs: np.ndarray,
             ci_lo: float, ci_hi: float,
             save_path: str = "ppo_vs_ecf_results1.png"):

    COLORS = {"PPO": "#378ADD", "PPO+ECF": "#D85A30"}
    phases = ["phase1", "phase2", "phase3", "phase4"]
    phase_labels = [
        "P1\nStandard training",
        "P2\nRetention",
        "P3\nDisturbance (wind)",
        "P4\nRecovery",
    ]
    phase_colors = ["#F0F4FA", "#F0F4FA", "#FFF3EC", "#F0F8F2"]

    fig = plt.figure(figsize=(22, 15))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.38)

    # ── Row 0: learning curves per phase ──
    for col, (phase, pl, bg) in enumerate(zip(phases, phase_labels, phase_colors)):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(bg)
        for agent in ["PPO", "PPO+ECF"]:
            seed_curves = []
            for seed_data in raw_per_seed[agent]:
                ep_rews = seed_data[phase]
                if ep_rews:
                    seed_curves.append(smooth(ep_rews, w=8))
            if not seed_curves:
                continue
            min_len = min(len(c) for c in seed_curves)
            mat  = np.stack([c[:min_len] for c in seed_curves])
            mean = mat.mean(0)
            std  = mat.std(0)
            x    = np.arange(len(mean))
            ax.plot(x, mean, color=COLORS[agent], linewidth=1.8, label=agent)
            ax.fill_between(x, mean - std, mean + std,
                            color=COLORS[agent], alpha=0.15)
        ax.set_title(pl, fontsize=10, fontweight="bold")
        ax.set_xlabel("episode", fontsize=9)
        ax.set_ylabel("reward", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    # ── Row 1: box plots per phase ──
    x_pos = [0, 1]
    for col, (phase, pl) in enumerate(zip(phases, phase_labels)):
        ax = fig.add_subplot(gs[1, col])
        data = [np.array(all_results[a][phase]) for a in ["PPO", "PPO+ECF"]]
        bp   = ax.boxplot(data, positions=x_pos, widths=0.45,
                          patch_artist=True, notch=False,
                          medianprops=dict(color="white", linewidth=2))
        for patch, agent in zip(bp["boxes"], ["PPO", "PPO+ECF"]):
            patch.set_facecolor(COLORS[agent])
            patch.set_alpha(0.75)
        for j, (d, agent) in enumerate(zip(data, ["PPO", "PPO+ECF"])):
            ax.scatter(np.random.normal(j, 0.05, len(d)), d,
                       s=20, color=COLORS[agent], alpha=0.6, zorder=5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(["PPO", "PPO+ECF"], fontsize=9)
        ax.set_title(f"{pl}\n(per-seed means)", fontsize=10, fontweight="bold")
        ax.set_ylabel("mean reward", fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    # ── Row 2 col 0-2: ECF internals ──
    ecf_seed_data = raw_per_seed["PPO+ECF"]
    I_all     = [x for s in ecf_seed_data for x in s["I_history"]]
    G_all     = [x for s in ecf_seed_data for x in s["G_history"]]
    bonus_all = [x for s in ecf_seed_data for x in s["bonus_history"]]

    def plot_internal(ax, data, title, color, ylabel):
        s = smooth(data, w=50)
        x = np.arange(len(s))
        ax.plot(x, s, color=color, linewidth=1.5)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("step", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    plot_internal(fig.add_subplot(gs[2, 0]), I_all,
                  "ECF: incoherence I", "#5DCAA5", "I")
    plot_internal(fig.add_subplot(gs[2, 1]), G_all,
                  "ECF: coherence G",   "#7F77DD", "G")
    plot_internal(fig.add_subplot(gs[2, 2]), bonus_all,
                  "ECF: intrinsic bonus", "#D85A30", "bonus")

    # ── Row 2 col 3: P3→P4 recovery gain + bootstrap CI ──
    ax_rec = fig.add_subplot(gs[2, 3])
    for j, agent in enumerate(["PPO", "PPO+ECF"]):
        p3 = np.array(all_results[agent]["phase3"])
        p4 = np.array(all_results[agent]["phase4"])
        delta = p4 - p3
        ax_rec.bar(j, delta.mean(), yerr=delta.std(),
                   color=COLORS[agent], width=0.5, capsize=5,
                   error_kw=dict(linewidth=1.5), alpha=0.8, label=agent)
        ax_rec.text(j, delta.mean() + delta.std() + 1.0,
                    f"{delta.mean():+.1f}", ha="center", fontsize=10,
                    color=COLORS[agent], fontweight="bold")
    ax_rec.axhline(0, color="#888", linewidth=0.8, linestyle="--")
    ax_rec.set_xticks([0, 1])
    ax_rec.set_xticklabels(["PPO", "PPO+ECF"], fontsize=9)
    ax_rec.set_title("Recovery gain\n(P4 − P3)", fontsize=10, fontweight="bold")
    ax_rec.set_ylabel("Δ reward", fontsize=9)
    ax_rec.grid(axis="y", alpha=0.25)
    ax_rec.spines[["top", "right"]].set_visible(False)

    # annotate bootstrap CI result
    ci_sig = "p<0.05" if (ci_lo > 0 or ci_hi < 0) else "ns"
    ax_rec.annotate(
        f"95% CI [{ci_lo:.1f}, {ci_hi:.1f}]\n{ci_sig}",
        xy=(0.5, 0.05), xycoords="axes fraction",
        ha="center", fontsize=8, color="#555",
    )

    fig.suptitle(
        "BipedalWalker-v3  |  PPO vs PPO+ECF  |  "
        f"{N_SEEDS} seeds  |  4-phase protocol",
        fontsize=14, fontweight="bold", y=0.995,
    )

    # phase boundary legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#F0F4FA", edgecolor="#aaa", label="Nominal env"),
        Patch(facecolor="#FFF3EC", edgecolor="#aaa", label="Wind disturbance"),
        Patch(facecolor="#F0F8F2", edgecolor="#aaa", label="Recovery"),
    ]
    fig.legend(handles=legend_elements, loc="upper right",
               fontsize=8, framealpha=0.8, ncol=3,
               bbox_to_anchor=(0.99, 0.99))

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {save_path}")


# ─────────────────────────── Main ──────────────────────────────────────

def main():
    global N_SEEDS, PHASE1_EPISODES, PHASE2_EPISODES, PHASE3_EPISODES, PHASE4_EPISODES, DISTURB_GRAVITY

    _n_seeds = N_SEEDS
    _p1      = PHASE1_EPISODES
    _p2      = PHASE2_EPISODES
    _p3      = PHASE3_EPISODES
    _p4      = PHASE4_EPISODES

    parser = argparse.ArgumentParser(
        description="PPO vs PPO+ECF on BipedalWalker-v3 (4-phase)")
    parser.add_argument("--seeds",           type=int,   default=_n_seeds)
    parser.add_argument("--phase1-episodes", type=int,   default=_p1)
    parser.add_argument("--phase2-episodes", type=int,   default=_p2)
    parser.add_argument("--phase3-episodes", type=int,   default=_p3)
    parser.add_argument("--phase4-episodes", type=int,   default=_p4)
    parser.add_argument("--disturb-gravity", type=float, default=DISTURB_GRAVITY)
    parser.add_argument("--ecf-latent-dim",  type=int,   default=24)
    parser.add_argument("--ecf-lam-max",     type=float, default=0.03)
    parser.add_argument("--ecf-bonus-scale", type=float, default=0.50)
    parser.add_argument("--plot-path",       type=str,   default="ppo_vs_ecf_results.png")
    args = parser.parse_args()

    N_SEEDS          = args.seeds
    PHASE1_EPISODES  = args.phase1_episodes
    PHASE2_EPISODES  = args.phase2_episodes
    PHASE3_EPISODES  = args.phase3_episodes
    PHASE4_EPISODES  = args.phase4_episodes
    DISTURB_GRAVITY  = args.disturb_gravity

    ecf_cfg = ECFConfig(
        latent_dim    = args.ecf_latent_dim,
        lam_max       = args.ecf_lam_max,
        novelty_scale = args.ecf_bonus_scale,
    )

    all_results = {
        "PPO":     {"phase1": [], "phase2": [], "phase3": [], "phase4": []},
        "PPO+ECF": {"phase1": [], "phase2": [], "phase3": [], "phase4": []},
    }
    raw_per_seed = {"PPO": [], "PPO+ECF": []}

    for seed_idx in range(N_SEEDS):
        seed = seed_idx * 42 + 1
        print(f"\n── Seed {seed_idx + 1}/{N_SEEDS}  (seed={seed}) ──")

        for agent_name, use_ecf in [("PPO", False), ("PPO+ECF", True)]:
            cfg_i = ECFConfig(**{**ecf_cfg.__dict__, "seed": seed})
            t0    = time.time()
            res   = run_seed(seed, use_ecf, cfg_i)
            dt    = time.time() - t0

            for phase in ["phase1", "phase2", "phase3", "phase4"]:
                m = last_n_mean(res[phase], 50)
                all_results[agent_name][phase].append(m)

            raw_per_seed[agent_name].append(res)
            p1 = last_n_mean(res["phase1"])
            p2 = last_n_mean(res["phase2"])
            p3 = last_n_mean(res["phase3"])
            p4 = last_n_mean(res["phase4"])
            print(f"  {agent_name:8s}  P1={p1:7.1f}  P2={p2:7.1f}  "
                  f"P3={p3:7.1f}  P4={p4:7.1f}  ({dt:.0f}s)")

    diffs, lo, hi, _ = print_stats(all_results)

    serialisable = {
        a: {p: [float(x) for x in v] for p, v in d.items()}
        for a, d in all_results.items()
    }
    with open("ppo_vs_ecf_results.json", "w") as f:
        json.dump(serialisable, f, indent=2)
    print("Raw results saved to ppo_vs_ecf_results.json")

    plot_all(all_results, raw_per_seed, diffs, lo, hi, args.plot_path)


if __name__ == "__main__":
    main()
