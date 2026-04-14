# merlion_finetuning_general.py

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import gymnasium as gym

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig


# ========= Helpers (kept intentionally close to sb3_independent) =========

def project_simplex(w: np.ndarray, min_weight: float = 1e-6) -> np.ndarray:
    # Ensure writable copy (Ray can deserialize arrays as read-only views)
    w = np.array(w, dtype=np.float32, copy=True)
    w[w < 0] = 0.0
    s = float(w.sum())
    if s <= 0:
        w[:] = 1.0 / float(w.size)
    else:
        w /= s

    w = np.maximum(w, float(min_weight))
    w /= float(w.sum())
    return w.astype(np.float32)


def sample_local_weights(w_base: np.ndarray, m: int, eps: float) -> List[np.ndarray]:
    outs: List[np.ndarray] = []
    w_base = np.asarray(w_base, dtype=np.float32).reshape(-1)
    for _ in range(int(m)):
        delta = np.random.uniform(-eps, eps, size=w_base.shape).astype(np.float32)
        outs.append(project_simplex(w_base + delta))
    return outs


def _unwrap_env_once(env: Any) -> Optional[Any]:
    for attr in ("unwrapped", "env", "base_env", "_env", "wrapped_env"):
        if hasattr(env, attr):
            nxt = getattr(env, attr)
            if callable(nxt):
                nxt = nxt()
            if nxt is not None and nxt is not env:
                return nxt
    return None


def _find_weight_setter_env(env: Any, max_hops: int = 16) -> Optional[Any]:
    """Walk wrappers until an object exposing set_scalarization_weights is found."""
    cur = env
    seen = set()
    for _ in range(int(max_hops)):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))

        if hasattr(cur, "set_scalarization_weights"):
            return cur

        # VecEnv-like containers
        if hasattr(cur, "envs") and getattr(cur, "envs"):
            try:
                cur = cur.envs[0]
                continue
            except Exception:
                pass

        nxt = _unwrap_env_once(cur)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    return None


def set_env_weights(env: Any, w: np.ndarray):
    """Push weights into env; supports either a setter or direct attribute."""
    w = project_simplex(np.asarray(w, np.float32))
    base = _find_weight_setter_env(env)
    if base is not None:
        base.set_scalarization_weights(w.tolist())
        return

    # fallback: common conventions
    if hasattr(env, "envs") and env.envs:
        for e in env.envs:
            b = _find_weight_setter_env(e)
            if b is not None:
                b.set_scalarization_weights(w.tolist())
            else:
                setattr(e, "weights", w.tolist())
    else:
        setattr(env, "weights", w.tolist())


def _infer_reward_dim_from_env(env_id: str, info_key: str = "mo_reward", seed: int = 0) -> Optional[int]:
    """Probe a fresh env once to infer vector reward dimension."""
    env = gym.make(env_id)
    try:
        env.reset(seed=int(seed))
        a = env.action_space.sample()
        _, _, terminated, truncated, info = env.step(a)
        _ = (terminated, truncated)

        if isinstance(info, dict):
            v = info.get(info_key)
            if v is None:
                v = info.get("mo_reward_raw")
            if v is not None:
                return int(np.asarray(v).reshape(-1).shape[0])

        # fallback to unwrapped.vector_reward if present
        try:
            u = env.unwrapped
            if hasattr(u, "vector_reward"):
                return int(np.asarray(getattr(u, "vector_reward")).reshape(-1).shape[0])
        except Exception:
            pass

        return None
    finally:
        try:
            env.close()
        except Exception:
            pass


def _pad_or_truncate(v: np.ndarray, M: int) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    if v.size < M:
        v = np.pad(v, (0, M - v.size), constant_values=0.0)
    elif v.size > M:
        v = v[:M]
    return v.astype(np.float32)


def _get_vec_from_info(env: gym.Env, info: Any, info_key: str) -> np.ndarray:
    v = None
    if isinstance(info, dict):
        v = info.get(info_key)
        if v is None:
            v = info.get("mo_reward_raw")

    if v is None:
        # final fallback: env.unwrapped.vector_reward if it exists
        try:
            u = env.unwrapped
            if hasattr(u, "vector_reward"):
                v = getattr(u, "vector_reward")
        except Exception:
            v = None

    if v is None:
        raise KeyError(
            f"Vector reward not found (missing info['{info_key}'] and 'mo_reward_raw' and env.unwrapped.vector_reward)."
        )

    return np.asarray(v, dtype=np.float32).reshape(-1)


def eval_vectorised_rllib(
    algo: Algorithm,
    env_id: str,
    w: np.ndarray,
    n_eval_episodes: int = 10,
    info_key: str = "mo_reward",
    seed: int = 0,
) -> Tuple[float, float, np.ndarray]:
    """Evaluate scalar return and vector return using the RLlib policy.

    Vector return is defined as the sum over the episode of info[info_key]
    (fallback: info['mo_reward_raw'], then env.unwrapped.vector_reward).

    Note: this evaluation environment is created via gym.make(env_id). Make sure
    env_id matches the environment used to train the checkpoint.
    """
    policy = algo.get_policy()

    scalar_returns: List[float] = []
    vec_returns: List[np.ndarray] = []

    env = gym.make(env_id)
    try:
        set_env_weights(env, w)

        for ep in range(int(n_eval_episodes)):
            obs, _ = env.reset(seed=int(seed + 10_000 + ep))
            done = False
            ep_scalar = 0.0
            ep_vec: Optional[np.ndarray] = None

            while not done:
                # compute_single_action returns (action, state_out, extra)
                act, _, _ = policy.compute_single_action(obs, explore=False)
                obs, r, terminated, truncated, info = env.step(act)
                ep_scalar += float(r)

                v = _get_vec_from_info(env, info, info_key)
                ep_vec = v.copy() if ep_vec is None else (ep_vec + v)

                done = bool(terminated or truncated)

            scalar_returns.append(ep_scalar)
            vec_returns.append(ep_vec if ep_vec is not None else np.zeros(1, dtype=np.float32))

        scalar_arr = np.asarray(scalar_returns, dtype=np.float32)
        vec_arr = np.stack(vec_returns, axis=0)
        return float(np.mean(scalar_arr)), float(np.std(scalar_arr)), np.mean(vec_arr, axis=0).astype(np.float32)

    finally:
        try:
            env.close()
        except Exception:
            pass


def _set_algo_env_weights(algo: Algorithm, w: np.ndarray):
    """Set scalarization weights on all rollout workers' envs."""

    w = project_simplex(np.asarray(w, np.float32))

    def _set_on_worker(worker):
        worker.foreach_env(lambda env: set_env_weights(env, w))

    # RLlib local + remote rollout workers
    if hasattr(algo, "workers") and algo.workers is not None:
        algo.workers.foreach_worker(_set_on_worker)

    # Evaluation workers (if configured)
    if hasattr(algo, "evaluation_workers") and algo.evaluation_workers is not None:
        algo.evaluation_workers.foreach_worker(_set_on_worker)


def _maybe_seed_algo_envs(algo: Algorithm, seed: int):
    """Best-effort seeding of envs across workers (does not guarantee full determinism)."""

    def _seed_on_worker(worker):
        def _seed_env(env):
            try:
                env.reset(seed=int(seed))
            except TypeError:
                try:
                    env.reset()
                except Exception:
                    pass
        worker.foreach_env(_seed_env)

    if hasattr(algo, "workers") and algo.workers is not None:
        algo.workers.foreach_worker(_seed_on_worker)


def _train_for_at_least_steps(algo: Algorithm, target_steps: int) -> Dict[str, Any]:
    """Call algo.train() until timesteps_total increases by at least target_steps."""
    target_steps = int(target_steps)
    if target_steps <= 0:
        return {}

    # Prefer reading the internal counter before training so we count *this* interval correctly.
    start_total = 0
    try:
        start_total = int(getattr(algo, "_counters", {}).get("timesteps_total", 0))
    except Exception:
        start_total = 0

    res: Dict[str, Any] = {}
    while True:
        res = algo.train()

        # If timesteps_total isn't present (older configs), fall back to a single train call.
        if "timesteps_total" not in res:
            return res

        cur_total = int(res.get("timesteps_total", 0))
        if cur_total - start_total >= target_steps:
            return res


def _discover_population(
    pop_dir: str,
    num_objectives: Optional[int],
    env_id_for_fallback: Optional[str] = None,
    info_key: str = "mo_reward",
    seed: int = 0,
) -> Tuple[List[Tuple[str, np.ndarray, int]], int]:
    """Return (population, M).

    population = list of (ck_dir, base_weight, idx).

    M inference order (to match the SB3-generalised behaviour):
      1) num_objectives if provided
      2) max length across all available weight.json
      3) env probe (info[info_key] / mo_reward_raw / unwrapped.vector_reward)
      4) fallback to 3
    """
    root = Path(pop_dir)
    dirs = sorted(
        root.glob("checkpoint_policy_*"),
        key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else 0,
    )

    weight_by_idx: Dict[int, np.ndarray] = {}
    max_dim = 0
    ck_by_idx: Dict[int, str] = {}

    for d in dirs:
        try:
            idx = int(d.name.split("_")[-1])
        except Exception:
            continue

        ck = d / "checkpoint_000000"
        if not ck.is_dir():
            continue
        ck_by_idx[idx] = str(ck)

        sidecar = d / "weight.json"
        if sidecar.exists():
            with sidecar.open("r") as f:
                data = json.load(f)
            w = data
            if isinstance(data, dict):
                w = data.get("weight") or data.get("merlion_weight") or data.get("weights")
            if w is not None:
                w_arr = np.asarray(w, dtype=np.float32).reshape(-1)
                weight_by_idx[idx] = w_arr
                max_dim = max(max_dim, int(w_arr.size))

    if num_objectives is not None:
        M = int(num_objectives)
    elif max_dim > 0:
        M = int(max_dim)
    else:
        M_probe = None
        if env_id_for_fallback is not None:
            M_probe = _infer_reward_dim_from_env(env_id_for_fallback, info_key=info_key, seed=seed)
        M = int(M_probe) if M_probe is not None else 3

    out: List[Tuple[str, np.ndarray, int]] = []
    for idx in sorted(ck_by_idx.keys()):
        ck_dir = ck_by_idx[idx]
        w = weight_by_idx.get(idx)
        if w is None:
            w = np.ones(M, dtype=np.float32) / float(M)
            print(f"[warn] No weight.json for policy {idx}; using uniform {w.tolist()}.")
        else:
            w = _pad_or_truncate(w, M)

        out.append((str(ck_dir), project_simplex(w), int(idx)))

    if not out:
        raise RuntimeError(f"No checkpoint_policy_* folders under {pop_dir}")

    return out, M

def _build_ppo_from_meta_ckpt(meta_ckpt_dir: str, env_id: str, env_config: dict | None = None) -> Algorithm:
    # 1) Load meta algo ONLY to grab weights (not full state)
    meta = Algorithm.from_checkpoint(meta_ckpt_dir)
    try:
        meta_pol = meta.get_policy()
        meta_weights = meta_pol.get_weights()   # <-- IMPORTANT: weights only

        # carry over model settings so architectures match
        model_cfg = dict(getattr(meta.config, "model", {}) or {})
        framework = getattr(meta.config, "framework_str", "torch")
        num_workers = int(getattr(meta.config, "num_rollout_workers", 0))
    finally:
        meta.stop()

    # 2) Build fresh PPO
    ppo = (
        PPOConfig()
        .environment(env=env_id, env_config=env_config or {})
        .framework(framework)
        .rollouts(num_rollout_workers=num_workers)
        .training(model=model_cfg)
        .build()
    )

    # 3) Inject meta weights without touching PPO config
    ppo.get_policy().set_weights(meta_weights)
    return ppo

# ========= Main fine-tuning (Algorithm 3) =========

def finetune_with_local_perturbations(
    pop_dir: str,
    env_id: str,
    out_dir: str,
    total_steps: int = 5000,
    record_every: int = 500,
    eval_episodes: int = 20,
    m_perturb: int = 5,
    eps: float = 0.05,
    seed: int = 7,
    num_objectives: Optional[int] = None,
    info_key: str = "mo_reward",
    skip_existing: bool = True,
    ray_init_kwargs: Optional[Dict[str, Any]] = None,
):
    """Fine-tune MERLION meta-policies with RLlib (local perturbations).

    If skip_existing=True and an offspring directory already exists, the code:
      - loads weight.json + learning_curve.csv
      - reconstructs the final snapshot
      - appends it into final_rows and hist_rows

    Output files under out_dir (same as independent script):
      - all_offspring_summary.csv
      - all_timesteps_long.csv

    Each offspring_dir contains:
      - weight.json
      - learning_curve.csv
      - a saved RLlib checkpoint (created by algo.save(...))
    """

    np.random.seed(int(seed))
    os.makedirs(out_dir, exist_ok=True)

    if not ray.is_initialized():
        kwargs = {"ignore_reinit_error": True, "include_dashboard": False}
        if ray_init_kwargs:
            kwargs.update(ray_init_kwargs)
        ray.init(**kwargs)

    population, M = _discover_population(
        pop_dir,
        num_objectives=num_objectives,
        env_id_for_fallback=env_id,
        info_key=info_key,
        seed=seed,
    )

    total_intervals = int(total_steps // record_every)
    if total_intervals <= 0:
        raise ValueError("total_steps must be >= record_every")

    base_out = Path(out_dir).resolve()
    base_out.mkdir(parents=True, exist_ok=True)

    final_rows: List[Dict[str, Any]] = []
    hist_rows: List[Dict[str, Any]] = []

    for ck_dir, w_i, idx in population:
        print(f"\n=== Meta-policy {idx} ===")

        # 3) Local perturbations around w_i
        W_local = [w_i] + sample_local_weights(w_i, m_perturb, eps)
        tags = [f"i{idx:03d}_m-1"] + [f"i{idx:03d}_m{m:02d}" for m in range(m_perturb)]

        for j, (w_im, offspring_tag) in enumerate(zip(W_local, tags)):
            m = -1 if j == 0 else j - 1
            offspring_dir = base_out / offspring_tag

            # ===================== RESUME / SKIP LOGIC =====================
            if skip_existing and offspring_dir.is_dir():
                print(
                    f"[skip+load] {offspring_tag}: {offspring_dir} already exists, "
                    "loading existing results into summaries."
                )

                # Try to load weight
                try:
                    with open(offspring_dir / "weight.json", "r") as f:
                        w_loaded = np.array(json.load(f), dtype=float).reshape(-1)
                except FileNotFoundError:
                    print("  [warn] weight.json missing, using w_im from schedule.")
                    w_loaded = np.asarray(w_im, dtype=float).reshape(-1)

                w_loaded = _pad_or_truncate(w_loaded, M)

                # Try to load learning curve
                lc_path = offspring_dir / "learning_curve.csv"
                if lc_path.exists():
                    df_lc = pd.read_csv(lc_path)
                    if len(df_lc) == 0:
                        print("  [warn] learning_curve.csv empty; skipping offspring in summary.")
                        continue

                    last = df_lc.iloc[-1]
                    t_idx = len(df_lc) - 1
                    timesteps = (t_idx + 1) * record_every

                    row = {
                        "i": idx,
                        "m": m,
                        "tag": offspring_tag,
                        "t_idx": t_idx,
                        "timesteps": timesteps,
                        "mean": float(last["mean"]),
                        "std": float(last["std"]),
                    }
                    for k in range(M):
                        row[f"w{k}"] = float(w_loaded[k])
                        col = f"reward_{k}"
                        row[col] = float(last[col]) if col in df_lc.columns else float("nan")
                    hist_rows.append(row)

                    final_row = {
                        "i": idx,
                        "m": m,
                        "tag": offspring_tag,
                        "offspring_dir": str(offspring_dir),
                        "mean_last": float(last["mean"]),
                        "elapsed_s": float("nan"),
                    }
                    for k in range(M):
                        final_row[f"w{k}"] = float(w_loaded[k])
                        col = f"reward_{k}"
                        final_row[f"reward_{k}_last"] = float(last[col]) if col in df_lc.columns else float("nan")
                    final_rows.append(final_row)

                else:
                    print("  [warn] learning_curve.csv not found; offspring not added to summary.")

                continue
            # =================================================================

            # NEW offspring: train
            offspring_dir.mkdir(parents=True, exist_ok=True)

            # Make the seed deterministic per (meta-policy idx, offspring m)
            seed_offspring = int(seed + idx * 1000 + (m + 1))

            # Load a fresh algo for THIS offspring (independent training)
            algo = _build_ppo_from_meta_ckpt(ck_dir, env_id=env_id)
            try:
                _maybe_seed_algo_envs(algo, seed_offspring)
                _set_algo_env_weights(algo, w_im)

                mean_hist = np.zeros(total_intervals, dtype=np.float32)
                std_hist = np.zeros(total_intervals, dtype=np.float32)
                rewards_hist = np.zeros((total_intervals, M), dtype=np.float32)

                t0 = time.time()
                mean_r = std_r = 0.0
                r_vec = np.zeros(M, dtype=np.float32)

                for t in range(total_intervals):
                    _train_for_at_least_steps(algo, record_every)

                    mean_r, std_r, r_vec_raw = eval_vectorised_rllib(
                        algo,
                        env_id=env_id,
                        w=w_im,
                        n_eval_episodes=eval_episodes,
                        info_key=info_key,
                        seed=seed_offspring + t * 123,
                    )
                    r_vec = _pad_or_truncate(r_vec_raw, M)

                    mean_hist[t] = float(mean_r)
                    std_hist[t] = float(std_r)
                    rewards_hist[t, :] = r_vec

                    print(f"     [{t+1}/{total_intervals}] R={mean_r:.2f}")

                # We only store final t in the long table (same as independent)
                t_idx = total_intervals - 1
                hist_row = {
                    "i": idx,
                    "m": m,
                    "tag": offspring_tag,
                    "t_idx": t_idx,
                    "timesteps": (t_idx + 1) * record_every,
                    "mean": float(mean_r),
                    "std": float(std_r),
                }
                for k in range(M):
                    hist_row[f"w{k}"] = float(w_im[k])
                    hist_row[f"reward_{k}"] = float(r_vec[k])
                hist_rows.append(hist_row)

                # Save algo checkpoint + weight
                ckpt_path = algo.save(str(offspring_dir))
                with open(offspring_dir / "checkpoint_path.txt", "w") as f:
                    f.write(str(ckpt_path))

                with open(offspring_dir / "weight.json", "w") as f:
                    json.dump([float(x) for x in np.asarray(w_im).reshape(-1)], f)

                # Save the full learning curve for this offspring
                lc: Dict[str, Any] = {"mean": mean_hist, "std": std_hist}
                for k in range(M):
                    lc[f"reward_{k}"] = rewards_hist[:, k]
                pd.DataFrame(lc).to_csv(offspring_dir / "learning_curve.csv", index=False)

                final_row = {
                    "i": idx,
                    "m": m,
                    "tag": offspring_tag,
                    "offspring_dir": str(offspring_dir),
                    "mean_last": float(mean_hist[-1]),
                    "elapsed_s": float(time.time() - t0),
                }
                for k in range(M):
                    final_row[f"w{k}"] = float(w_im[k])
                    final_row[f"reward_{k}_last"] = float(rewards_hist[-1, k])
                final_rows.append(final_row)

            finally:
                try:
                    algo.stop()
                except Exception:
                    pass

    # Save combined summaries across ALL offspring (old + new)
    pd.DataFrame(final_rows).to_csv(base_out / "all_offspring_summary.csv", index=False)
    pd.DataFrame(hist_rows).to_csv(base_out / "all_timesteps_long.csv", index=False)

    print(
        "Saved:\n"
        f"- {base_out/'all_offspring_summary.csv'}\n"
        f"- {base_out/'all_timesteps_long.csv'}\n"
    )


__all__ = [
    "finetune_with_local_perturbations",
    "project_simplex",
    "sample_local_weights",
    "set_env_weights",
    "eval_vectorised_rllib",
]
