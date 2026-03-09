"In this version, multipe tasks are sampled within a inner loop batch."
# merlion_fixed.py
# MERLION: Meta Multi-Objective RL with Evolutionary Technique
# This file fixes reward breakdown and aligns Algorithms 1, 2, 3 with the proposal.

import sys, os
sys.path.append("/home/rifnyrachman7/_metamorl")

import logging
import numpy as np
from typing import Optional, Type, List, Tuple, Dict, Any
import random
import copy

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.evaluation.metrics import get_learner_stats
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.common import (
    STEPS_SAMPLED_COUNTER,
    STEPS_TRAINED_COUNTER,
    STEPS_TRAINED_THIS_ITER_COUNTER,
    _get_shared_metrics,
)
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import (
    concat_samples,
    convert_ma_batch_to_sample_batch,
    SampleBatch,
)
from ray.rllib.execution.metric_ops import CollectMetrics
from ray.rllib.evaluation.metrics import collect_metrics
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import Deprecated, DEPRECATED_VALUE
from ray.rllib.utils.metrics.learner_info import LEARNER_INFO
from ray.rllib.utils.sgd import standardized
from ray.util.iter import from_actors, LocalIterator
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch, concat_samples

from .merlion_utils import *
from .merlion_archive import MERLIONArchive

logger = logging.getLogger(__name__)

# ==========================
# pushing bask the weights
# ==========================

import ray
import numpy as np
import inspect

def _unwrap_env_once(e):
    # Common wrapper attributes
    for attr in ("unwrapped", "env", "base_env", "_env", "wrapped_env"):
        if hasattr(e, attr):
            try:
                nxt = getattr(e, attr)
                if callable(nxt):  # some libs expose unwrapped()
                    nxt = nxt()
                if nxt is not None and nxt is not e:
                    return nxt
            except Exception:
                pass
    return None

def _find_base_env(e, max_hops=16):
    """Walk through typical RLlib/Gym wrappers to reach the user env."""
    seen = set()
    cur = e
    for _ in range(max_hops):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        # Found the method on this layer?
        if hasattr(cur, "set_scalarization_weights"):
            return cur
        # Vector envs sometimes carry a single sub-env in `.envs` or `get_sub_environments()`
        if hasattr(cur, "envs") and isinstance(cur.envs, (list, tuple)) and cur.envs:
            cur = cur.envs[0]
            continue
        if hasattr(cur, "get_sub_environments"):
            try:
                subs = cur.get_sub_environments()
                if subs:
                    cur = subs[0]
                    continue
            except Exception:
                pass
        nxt = _unwrap_env_once(cur)
        if nxt is None:
            break
        cur = nxt
    return None

def _safe_set_w(e, w):
    """Reach the base env and call set_scalarization_weights(w)."""
    w = (np.asarray(w, dtype=np.float32) / max(np.sum(w), 1e-12)).tolist()
    base = _find_base_env(e)
    if base is None or not hasattr(base, "set_scalarization_weights"):
        # Debug aid: print wrapper chain (once) for this actor
        try:
            chain = []
            cur = e; hops = 0
            while cur is not None and hops < 16:
                chain.append(type(cur).__name__)
                nxt = _unwrap_env_once(cur)
                if nxt is None or nxt is cur: break
                cur = nxt; hops += 1
            print("[MERLION] Could not find set_scalarization_weights. Wrapper chain:", " -> ".join(chain))
        except Exception:
            pass
        raise AttributeError("Env (or its wrappers) has no set_scalarization_weights(w)")
    base.set_scalarization_weights(w)
    return True

def _foreach_env_call(rw, fn):
    """Call foreach_env correctly on local OR remote rollout workers."""
    fe = rw.foreach_env
    return ray.get(fe.remote(fn)) if hasattr(fe, "remote") else fe(fn)

def _infer_num_objectives_from_workers(algo, default: int = 2) -> int:
    """
    Try to infer the objective vector dimension D by peeking at the first
    trajectory's info dict. Falls back to `default` if nothing is found.
    """
    keys = ("mo_reward_vec", "mo_reward", "r_vec_used", "r_vec_oriented", "r_vec_raw")

    def _extract_D_from_batch(batch) -> int | None:
        # Handle both MA and SA batches
        try:
            from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
        except Exception:
            # If imports fail for any reason, just bail out
            return None

        sb = None
        if isinstance(batch, MultiAgentBatch):
            # Pick the first policy's SampleBatch
            if batch.policy_batches:
                sb = next(iter(batch.policy_batches.values()))
        elif isinstance(batch, SampleBatch):
            sb = batch

        if sb is None:
            return None

        infos = sb.get(SampleBatch.INFOS, None) or sb.get("infos", None)
        if not infos:
            return None

        for inf in infos:
            if isinstance(inf, dict):
                for k in keys:
                    if k in inf:
                        v = np.asarray(inf[k])
                        if v.ndim == 1 and v.size >= 1:
                            return int(v.size)
        return None

    # 1) Try local worker first (fast path)
    try:
        lw = algo.workers.local_worker()
        batch = lw.sample()
        D = _extract_D_from_batch(batch)
        if D is not None:
            return D
    except Exception:
        pass

    # 2) Fallback: ask a remote worker
    try:
        outs = algo.workers.foreach_worker(lambda w: w.sample())
        for b in outs:
            D = _extract_D_from_batch(b)
            if D is not None:
                return D
    except Exception:
        pass

    # 3) Give up: use default
    return int(default)


def _coerce_weight_dim(w, D: int) -> np.ndarray:
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    v = np.zeros(D, dtype=np.float32)
    n = min(D, w.size)
    if n > 0:
        v[:n] = w[:n]
    s = float(v.sum())
    if s <= 0.0:
        v[:] = 1.0 / max(D, 1)
    else:
        v /= s
    return v

# ==================
# RLlib Callbacks
# ==================

# class MERLIONCallbacks(DefaultCallbacks):
#     def on_postprocess_trajectory(
#         self, *, worker, episode, agent_id, policy_id, policies,
#         postprocessed_batch: SampleBatch, original_batches, **kwargs
#     ):
#         D = _infer_num_objectives_from_workers(workers, fallback=2) # get number of objectives
#         try:
#             # 1) Get the first mapping value from original_batches
#             first_val = next(iter(original_batches.values()))

#             # 2) Find the SampleBatch inside it:
#             #    - In newer RLlib: (policy, batch) or (policy, batch, agent_id)
#             #    - In some paths, value may already be a SampleBatch
#             orig_batch = None
#             if isinstance(first_val, (list, tuple)):
#                 for x in first_val:
#                     if isinstance(x, SampleBatch):
#                         orig_batch = x
#                         break
#             elif isinstance(first_val, SampleBatch):
#                 orig_batch = first_val

#             if orig_batch is None:
#                 # Nothing we can do without the original SampleBatch
#                 return

#             # 3) Pull per-step infos robustly
#             infos = orig_batch.get(SampleBatch.INFOS, None)
#             if infos is None:
#                 # Some stacks store it under "infos"
#                 infos = orig_batch.get("infos", None)
#             if infos is None:
#                 return

#             # Guard length match
#             if len(infos) != len(postprocessed_batch):
#                 # Different lengths: don't attempt to align step-wise
#                 return

#             # 4) Extract mo_reward from each step's info (default to zeros)
#             mo = []
#             for info in infos:
#                 if isinstance(info, dict) and ("mo_reward" in info):
#                     v = info["mo_reward"]
#                 else:
#                     v = [_ for _ in range(D)]
#                 mo.append(v)
#             mo = np.asarray(mo, dtype=np.float32)

#             # 5) Write per-step fields used by the trainer
#             postprocessed_batch["mo_reward_vec"] = mo
#             for d in range(D):
#                 postprocessed_batch[f"rewards-{d}"] = mo[:, d]

#         except Exception as e:
#             logger.error(f"MERLIONCallbacks error: {e}")
class MERLIONCallbacks(DefaultCallbacks):
    def on_postprocess_trajectory(
        self, *, worker, episode, agent_id, policy_id, policies,
        postprocessed_batch: SampleBatch, original_batches, **kwargs
    ):
        # try:
        # 1) Pull the original SampleBatch back out
        first_val = next(iter(original_batches.values()))
        if isinstance(first_val, (list, tuple)):
            orig_batch = next((x for x in first_val if isinstance(x, SampleBatch)), None)
        else:
            orig_batch = first_val if isinstance(first_val, SampleBatch) else None
        if orig_batch is None:
            return

        # 2) Get per-step infos (handle different RLlib keys)
        infos = orig_batch.get(SampleBatch.INFOS, None)
        if infos is None:
            infos = orig_batch.get("infos", None)
        if infos is None:
            return
        if len(infos) != len(postprocessed_batch):
            # lengths out of sync -> don't try to align step-wise
            return

        # 3) Infer D from the first info that has a vector; else fall back to config or 2
        D = None
        for inf in infos:
            if isinstance(inf, dict) and ("mo_reward" in inf):
                v = np.asarray(inf["mo_reward"])
                if v.ndim == 1 and v.size >= 1:
                    D = int(v.size); break
        if D is None:
            D = int(getattr(worker, "policy_config", {}).get("num_objectives", 2))

        # 4) Build dense [T, D] matrix, filling missing with zeros(D)
        mo = []
        raw = []
        #raise Exception(infos)
        for inf in infos:
            if isinstance(inf, dict) and ("mo_reward" in inf):
                vec = np.asarray(inf["mo_reward"], dtype=np.float32).reshape(-1)
                if vec.size != D:
                    # pad or trim to D
                    vv = np.zeros(D, np.float32); vv[:min(D, vec.size)] = vec[:D]
                    vec = vv
                mo.append(vec)
                raw.append(np.asarray(inf.get("mo_reward_raw"), dtype=np.float32).reshape(-1)
                           if "mo_reward_raw" in inf else None)
            else:
                mo.append(np.zeros(D, dtype=np.float32))
                raw.append(None)
        mo = np.stack(mo, axis=0).astype(np.float32, copy=False)

        # (optional) debug prints
        wid = getattr(worker, "worker_index", -1)
        # print(f"[DBG][w{wid}] mo_reward mean per-dim: {mo.mean(axis=0).tolist()} (shape={mo.shape})")

        # 5) Expose fields
        postprocessed_batch["mo_reward_vec"] = mo
        for d in range(D):
            postprocessed_batch[f"rewards-{d}"] = mo[:, d]
        # except Exception as e:
        #     logger.error(f"MERLIONCallbacks error: {e}")


# ==================
# Config
# ==================

class MERLIONConfig(AlgorithmConfig):
    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or MERLION)

        # Use PyTorch
        self.framework("torch")

        # --- Training / PPO+MAML knobs (single source of truth) ---
        self.training(
            # PPO-style
            use_gae=True,
            lambda_=1.0,
            kl_coeff=5e-4,
            vf_loss_coeff=0.5,
            entropy_coeff=0.0,
            clip_param=0.3,
            vf_clip_param=10.0,   # was 'svf_clip_param' (typo) — corrected
            grad_clip=1.0, # default None
            kl_target=0.01,

            # MAML-specific
            meta_batch_size=2,          # required for your (K,1) split
            inner_adaptation_steps=1,   # K = inner_adaptation_steps + 1
            inner_lr=0.1,
        )

        # --- Rollout/Env knobs ---
        self.num_rollout_workers = 2
        self.rollout_fragment_length = 200
        self.create_env_on_local_worker = True
        self.batch_mode = "complete_episodes"

        # Learning rate (outer optimizer)
        self.lr = 1e-3

        # Model
        self.model.update({"vf_share_layers": False})

        # Keep custom exec plan enabled
        self._disable_execution_plan_api = False

        # --- MERLION specifics ---
        self.use_meta_env = True
        self.maml_optimizer_steps = 5
        self.population_size = 10
        self.num_objectives = None

        # Init policy diversity
        self.init_method = "mixed"        # "seed" | "scale" | "mixed" | "pretrained" | "orthogonal" | "strategy"
        self.init_kwargs = {}             # extra args to your generators

        # Callbacks (to write mo_reward_vec etc.)
        self.callbacks_class = MERLIONCallbacks

        # Keep for backward-compat with older PPO configs that look for this flag
        self.vf_share_layers = DEPRECATED_VALUE

    def training(
        self, *,
        # PPO-ish
        use_gae: Optional[bool] = NotProvided,
        lambda_: Optional[float] = NotProvided,
        kl_coeff: Optional[float] = NotProvided,
        vf_loss_coeff: Optional[float] = NotProvided,
        entropy_coeff: Optional[float] = NotProvided,
        clip_param: Optional[float] = NotProvided,
        vf_clip_param: Optional[float] = NotProvided,
        grad_clip: Optional[float] = NotProvided,
        kl_target: Optional[float] = NotProvided,

        # MERLION/MAML custom knobs
        meta_batch_size: Optional[int] = NotProvided,          # <-- NEW: consume it here
        inner_adaptation_steps: Optional[int] = NotProvided,
        maml_optimizer_steps: Optional[int] = NotProvided,
        inner_lr: Optional[float] = NotProvided,
        use_meta_env: Optional[bool] = NotProvided,
        population_size: Optional[int] = NotProvided,
        num_objectives: Optional[int] = NotProvided,

        **kwargs,
    ) -> "MERLIONConfig":
        # Don't forward our custom keys to the base class:
        kwargs.pop("meta_batch_size", None)
        kwargs.pop("inner_adaptation_steps", None)
        kwargs.pop("maml_optimizer_steps", None)
        kwargs.pop("inner_lr", None)
        kwargs.pop("use_meta_env", None)
        kwargs.pop("population_size", None)
        kwargs.pop("num_objectives", None)

        # Let AlgorithmConfig handle the standard training keys
        super().training(**kwargs)

        # Store everything as attributes on this config
        if use_gae is not NotProvided: self.use_gae = use_gae
        if lambda_ is not NotProvided: self.lambda_ = lambda_
        if kl_coeff is not NotProvided: self.kl_coeff = kl_coeff
        if vf_loss_coeff is not NotProvided: self.vf_loss_coeff = vf_loss_coeff
        if entropy_coeff is not NotProvided: self.entropy_coeff = entropy_coeff
        if clip_param is not NotProvided: self.clip_param = clip_param
        if vf_clip_param is not NotProvided: self.vf_clip_param = vf_clip_param
        if grad_clip is not NotProvided: self.grad_clip = grad_clip
        if kl_target is not NotProvided: self.kl_target = kl_target

        if meta_batch_size is not NotProvided: self.meta_batch_size = meta_batch_size
        if inner_adaptation_steps is not NotProvided: self.inner_adaptation_steps = inner_adaptation_steps
        if maml_optimizer_steps is not NotProvided: self.maml_optimizer_steps = maml_optimizer_steps
        if inner_lr is not NotProvided: self.inner_lr = inner_lr
        if use_meta_env is not NotProvided: self.use_meta_env = use_meta_env
        if population_size is not NotProvided: self.population_size = population_size
        if num_objectives is not NotProvided: self.num_objectives = num_objectives
        return self


    @override(AlgorithmConfig)
    def validate(self) -> None:
        super().validate()
        if self.num_gpus > 1:
            raise ValueError("`num_gpus` > 1 not yet supported for MERLION!")
        if self.inner_adaptation_steps <= 0:
            raise ValueError("Inner Adaptation Steps must be >=1!")
        if self.maml_optimizer_steps <= 0:
            raise ValueError("PPO steps for meta-update needs to be >=0!")
        if self.entropy_coeff < 0:
            raise ValueError("`entropy_coeff` must be >=0.0!")
        if self.batch_mode != "complete_episodes":
            raise ValueError("`batch_mode`=truncate_episodes not supported!")
        if self.num_rollout_workers <= 0:
            raise ValueError("Must have at least 1 worker/task!")
        if self.create_env_on_local_worker is False:
            raise ValueError("Must create env on local worker (`create_env_on_local_worker=True`).")
        if self.population_size <= 0:
            raise ValueError("Population size must be > 0!")

# ==========================
# Task handling helpers
# ==========================

def set_worker_tasks(workers: WorkerSet, use_meta_env: bool):
    """Simple task refresh for SimpleSC: set_task() takes no args."""
    if not use_meta_env:
        return
    # Local worker
    workers.local_worker().foreach_env(lambda env: env.set_task())
    # Remote workers
    for w in workers.remote_workers():
        w.foreach_env.remote(lambda env: env.set_task())


# ==========================
# Meta-Update (Algorithm 1)
# ==========================

class MERLIONMetaUpdate:
    def __init__(self, workers, maml_steps, metric_gen, use_meta_env, archive: MERLIONArchive, worker_for_i):
        self.workers = workers
        self.maml_optimizer_steps = maml_steps
        self.metric_gen = metric_gen
        self.use_meta_env = use_meta_env
        self.archive = archive
        self.worker_for_i = worker_for_i

        # Defaults, read actual values from policy.config at runtime.
        self.gamma_default = 0.99
        self.lam_default = 1.0  ## debugging
        
    def _sanitize_samplebatch(self, sb: SampleBatch, clip_val: float = 1e6) -> SampleBatch:
        data = getattr(sb, "data", sb)
        for k, v in list(data.items()):
            if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
                arr = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                arr = np.clip(arr, -clip_val, clip_val)
                data[k] = arr.astype(v.dtype, copy=False)
        # Some RLlib stacks keep obs as dict
        if "obs" in data and isinstance(data["obs"], dict):
            for ok, ov in data["obs"].items():
                if isinstance(ov, np.ndarray) and np.issubdtype(ov.dtype, np.number):
                    arr = np.nan_to_num(ov, nan=0.0, posinf=0.0, neginf=0.0)
                    arr = np.clip(arr, -clip_val, clip_val)
                    data["obs"][ok] = arr.astype(ov.dtype, copy=False)
        return sb

    def _sample_from_worker(self, rw, episodes=1):
        gathered, done_eps, cap = [], 0, max(1, episodes)*6
        while done_eps < episodes and cap > 0:
            mb = ray.get(rw.sample.remote())
            sb = convert_ma_batch_to_sample_batch(mb) if isinstance(mb, MultiAgentBatch) else mb
            gathered.append(sb)
            cap -= 1
            d = sb.get(SampleBatch.DONES, None)
            if d is not None:
                done_eps += int(np.asarray(d, dtype=np.bool_).sum())
        return concat_samples(gathered) if len(gathered) > 1 else gathered[0]
    
    def __call__(self, data_tuple):
        _shared_samples = data_tuple[0]   # only for metrics passthrough
        adapt_metrics_dict = data_tuple[1]

        set_worker_tasks(self.workers, self.use_meta_env)
        lw = self.workers.local_worker()
        base_policy = lw.get_policy()

        # Read knobs
        gamma = float(base_policy.config.get("gamma", self.gamma_default))
        lam   = float(base_policy.config.get("lambda", self.lam_default))
        inner_steps = int(base_policy.config.get("inner_adaptation_steps", 1))
        K = inner_steps + 1

        # We always feed ONE task per i
        # (We’ll also push this down to each remote policy below.)
        J = int(base_policy.config.get("meta_batch_size", 1))
        base_policy.config["meta_batch_size"] = J
        base_policy.config["inner_adaptation_steps"] = inner_steps

        # --- Map each i to a rollout worker (round-robin if P > #workers)
        P = self.archive.population_size
        
        groups: Dict[Any, List[int]] = {}
        for i in range(P):
            rw = self.worker_for_i[i % len(self.worker_for_i)]
            groups.setdefault(rw, []).append(i)

        fetches_any = None

        # process workers in parallel; per-worker i's serially (to avoid weight races)
        for rw, idxs in groups.items():
            # Sync remote policy config + task once per worker.
            ray.get([
                rw.foreach_policy.remote(
                    lambda p, pid: (p.config.__setitem__("meta_batch_size", J),
                                    p.config.__setitem__("inner_adaptation_steps", inner_steps))
                ),
                rw.foreach_env.remote(lambda env: getattr(env, "set_task", lambda: None)())
            ])
            # Optional clamp for stability
            try:
                ray.get(self._clamp_free_log_std(rw))
            except Exception:
                pass

            for i in idxs:
                item = self.archive.get_archive_item(i)
                if item is None:
                    continue
                theta_i, w_i = item["theta"], item["weight"]

                # Push θᵢ to this worker.
                ray.get(rw.foreach_policy.remote(lambda p, pid, th=theta_i: p.set_weights(th)))

                _foreach_env_call(rw, lambda e, ww=w_i: _safe_set_w(e, ww))

                task_batches = []
                split_rows = []

                for j in range(J):
                    # (Re)sample a new task on this worker for task j
                    try:
                        ray.get(rw.foreach_env.remote(lambda env: getattr(env, "set_task", lambda: None)()))
                    except AttributeError:
                        # local worker path
                        rw.foreach_env(lambda env: getattr(env, "set_task", lambda: None)())

                    # Collect at least K done episodes for this task
                    # (need_eps=K ensures K episode boundaries are present for MAML)
                    batch_j = self._gather_batch(rw, min_steps=K, need_eps=K, cap=max(16, 8*K))

                    # Build per-task split of length K along episode boundaries
                    n_j = int(getattr(batch_j, "count", len(batch_j[SampleBatch.REWARDS])))
                    sizes_j = self._build_maml_split_on_episode_boundaries(batch_j, K)

                    # Fallback if anything is off
                    if sum(sizes_j) != n_j or any(s <= 0 for s in sizes_j):
                        sizes_j = self._equal_split(K, n_j).tolist()

                    task_batches.append(batch_j)
                    split_rows.append(np.asarray(sizes_j, dtype=np.int64))

                # Concatenate all J task batches in the same order we computed split_rows
                batch_i = concat_samples(task_batches) if len(task_batches) > 1 else task_batches[0]
                split = np.vstack(split_rows).reshape(J, K)
                batch_i["split"] = split

                # Scalarize + advantages
                cur = self._scalarize_and_make_advantages(batch_i, w_i, gamma, lam)

                # Ensure split integrity again after transforms
                sp = np.asarray(cur.get("split", split), dtype=np.int64)
                if sp.ndim == 1:
                    sp = sp.reshape(1, -1)
                elif sp.shape == (K, 1):
                    sp = sp.T
                # Check that sum matches batch length
                n_cur = int(getattr(cur, "count", len(cur[SampleBatch.REWARDS])))
                if sp.shape != (J, K) or int(sp.sum()) != n_cur or np.any(sp <= 0):
                    # Evenly divide by J×K as a last resort
                    sp = np.asarray(self._equal_split(J*K, n_cur), dtype=np.int64).reshape(J, K)
                cur["split"] = sp

                # Sanitize the full batch before training
                cur = self._sanitize_samplebatch(cur)

                # Inner optimizer steps (on this worker) — unchanged
                last_ref = None
                for _ in range(self.maml_optimizer_steps):
                    last_ref = rw.learn_on_batch.remote(cur)
                fetches = ray.get(last_ref) if last_ref is not None else None
                if fetches is not None:
                    fetches_any = fetches

                # Pull updated θᵢ back
                wlist = ray.get(rw.foreach_policy.remote(lambda p, pid: p.get_weights()))
                updated_theta = wlist[0] if isinstance(wlist, list) and len(wlist) > 0 else wlist
                #print("self.archive",self.archive.archive[0]["objectives"]) ## debugging

                updated_theta = self._sanitize_weights(updated_theta, theta_i)

                # ------------------------------------------------------------------
                # INSERTED: ensure env uses w_i BEFORE the evaluation sample, too
                _foreach_env_call(rw, lambda e, ww=w_i: _safe_set_w(e, ww))
                # ------------------------------------------------------------------

                # Evaluate this i on this worker
                eval_batch = self._gather_batch(rw, min_steps=K, need_eps=1, cap=8)
                obj_vals = self._extract_objective_values_from_samples(eval_batch)

                # Update archive for this i
                #print("objective values:",obj_vals)
                self.archive.update_archive_with_objectives(updated_theta, w_i, i, obj_vals)

        # Evolve
        self.archive.print_population_objectives()
        self.archive.evolve_weights()

        # learner stats
        if fetches_any is not None:
            learner_stats = get_learner_stats(fetches_any)
            def update(pi, pi_id):
                if pi_id in learner_stats and "inner_kl" in learner_stats[pi_id]:
                    pi.update_kls(learner_stats[pi_id]["inner_kl"])
            lw.foreach_policy_to_train(update)

        metrics = _get_shared_metrics()
        metrics.info[LEARNER_INFO] = fetches_any if fetches_any is not None else {}
        res = self.metric_gen.__call__(None)
        res.update(adapt_metrics_dict)
        return res


    def _sanitize_weights(self, w: Dict[str, np.ndarray], fallback: Dict[str, np.ndarray], clip_val: float = 1e3):
        cleaned, bad = {}, False
        for k, v in w.items():
            arr = np.asarray(v)
            if not np.all(np.isfinite(arr)):
                bad = True
                break
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            arr = np.clip(arr, -clip_val, clip_val)
            cleaned[k] = arr.astype(v.dtype, copy=False)
        return fallback if bad else cleaned

    def _collect_batches_parallel(self, i2worker: dict, K: int, need_eps: int = 1, cap: int = 16):
        """
        Collect 1 SampleBatch per solution-i in parallel from their mapped workers.
        Returns: dict {i: SampleBatch}
        """
        result = {}
        # Per-i progress
        prog = {
            i: {"steps": 0, "eps": 0, "cap": cap, "pieces": []}
            for i in i2worker.keys()
        }
        # Kick off one sample() per i
        fut_to_i = {}
        for i, rw in i2worker.items():
            f = rw.sample.remote()
            fut_to_i[f] = i

        while fut_to_i:
            ready, _ = ray.wait(list(fut_to_i.keys()), num_returns=1)
            f = ready[0]
            i = fut_to_i.pop(f)
            rw = i2worker[i]

            mb = ray.get(f)
            sb = convert_ma_batch_to_sample_batch(mb) if isinstance(mb, MultiAgentBatch) else mb
            prog[i]["pieces"].append(sb)

            data = getattr(sb, "data", sb)
            # steps
            n = int(getattr(sb, "count", len(data[SampleBatch.REWARDS])))
            prog[i]["steps"] += n
            # eps
            d = data.get(SampleBatch.DONES, None)
            if d is not None:
                prog[i]["eps"] += int(np.asarray(d, dtype=np.bool_).sum())
            else:
                term = data.get("terminateds", None); trunc = data.get("truncateds", None)
                if term is not None or trunc is not None:
                    term = np.asarray(term, dtype=np.bool_) if term is not None else 0
                    trunc = np.asarray(trunc, dtype=np.bool_) if trunc is not None else 0
                    prog[i]["eps"] += int((term | trunc).sum())
                else:
                    seq = data.get("seq_lens", None)
                    prog[i]["eps"] += int(np.asarray(seq).size) if seq is not None else 1

            prog[i]["cap"] -= 1
            done = ((prog[i]["steps"] >= K) and (prog[i]["eps"] >= need_eps)) or (prog[i]["cap"] <= 0)

            if done:
                pieces = prog[i]["pieces"]
                batch = pieces[0] if len(pieces) == 1 else concat_samples(pieces)
                result[i] = batch
            else:
                # ask for another chunk for this i
                nf = rw.sample.remote()
                fut_to_i[nf] = i

        return result
    
    def _evaluate_objectives_quick(self, episodes: int = 1) -> np.ndarray:
        lw = self.workers.local_worker()
        gathered: List[SampleBatch] = []
        done_eps = 0
        safety_cap = max(1, episodes) * 5

        while done_eps < episodes and safety_cap > 0:
            batch = lw.sample()
            sb = convert_ma_batch_to_sample_batch(batch) if isinstance(batch, MultiAgentBatch) else batch

            dones = sb.get(SampleBatch.DONES, None)
            if dones is not None:
                done_eps += int(np.asarray(dones, dtype=np.bool_).sum())
            else:
                term = sb.get("terminateds", None)
                trunc = sb.get("truncateds", None)
                if term is not None or trunc is not None:
                    term = np.asarray(term, dtype=np.bool_) if term is not None else 0
                    trunc = np.asarray(trunc, dtype=np.bool_) if trunc is not None else 0
                    done_eps += int((term | trunc).sum())
                else:
                    seq = sb.get("seq_lens", None)
                    if seq is not None:
                        done_eps += int(np.asarray(seq).size)
                    else:
                        done_eps += 1

            gathered.append(sb)
            safety_cap -= 1

        eval_batch = gathered[0] if len(gathered) == 1 else concat_samples(gathered)
        return self._extract_objective_values_from_samples(eval_batch)
    
    def _extract_objective_values_from_samples(self, samples) -> np.ndarray:
        data = getattr(samples, "data", samples)
        D = int(self.archive.num_objectives)  # get number of objectives
        keys = tuple(f"rewards-{d}" for d in range(D))

        rm = []
        has_all = all(k in data for k in keys)  # check once
        for d in range(D):
            if has_all:
                arr = np.asarray(data[f"rewards-{d}"], dtype=np.float32)
                rm_mean = float(np.mean(arr)) if arr.size else 0.0
                rm.append(rm_mean)

        # only return if we actually built the per-dimension means
        if has_all and len(rm) == D:
            return np.asarray(rm, dtype=np.float32)[:D]
        
        print('CHECK REWARD VEC DATA',data)

        if "mo_reward_vec" in data:
            # print('there is mo reward vec')
            R = np.asarray(data["mo_reward_vec"], dtype=np.float32)
            if R.ndim == 2 and R.shape[1] >= 1:
                d_use = min(D, R.shape[1])
                return R[:, :d_use].mean(axis=0).astype(np.float32)
        # else:
        #     print('NO MO REWARD VEC, STUPID')

        return np.zeros(D, dtype=np.float32)

    def _scalarize_samples_with_weight(self, samples, weight_vector):
        sb = copy.deepcopy(samples)
        data = getattr(sb, "data", sb)

        # 0) Scalarize reward
        if "mo_reward_vec" not in data:
            # Nothing we can do; return as-is
            return sb
        R = np.asarray(data["mo_reward_vec"], dtype=np.float32)
        r_scalar = R @ weight_vector[:R.shape[1]]
        data["rewards"] = r_scalar.astype(np.float32)

        # 1) Build robust dones
        T = len(r_scalar)
        dones = None

        # Primary: 'dones'
        if SampleBatch.DONES in data:
            dones = np.asarray(data[SampleBatch.DONES], dtype=np.bool_)

        # Gymnasium: 'terminateds'/'truncateds'
        if dones is None:
            term = data.get("terminateds", None)
            trunc = data.get("truncateds", None)
            if term is not None or trunc is not None:
                term = np.asarray(term, dtype=np.bool_) if term is not None else np.zeros(T, dtype=np.bool_)
                trunc = np.asarray(trunc, dtype=np.bool_) if trunc is not None else np.zeros(T, dtype=np.bool_)
                dones = np.asarray(term | trunc, dtype=np.bool_)

        # Sequence fallback: use 'seq_lens' to mark segment ends as terminal
        if dones is None:
            seq = data.get("seq_lens", None)
            if seq is not None:
                seq = np.asarray(seq, dtype=np.int32)
                dones = np.zeros(T, dtype=np.bool_)
                ends = np.cumsum(seq) - 1
                ends = ends[(ends >= 0) & (ends < T)]
                dones[ends] = True

        # Ultimate fallback: mark last step as done
        if dones is None:
            dones = np.zeros(T, dtype=np.bool_)
            dones[-1] = True

        # 2) Value baseline: 'vf_preds' or reconstruct or zeros
        vf = data.get(SampleBatch.VF_PREDS, None)
        if vf is None:
            vt = data.get("value_targets", None)
            adv0 = data.get("advantages", None)
            if vt is not None and adv0 is not None:
                vf = np.asarray(vt, dtype=np.float32) - np.asarray(adv0, dtype=np.float32)
            else:
                vf = np.zeros(T, dtype=np.float32)
        else:
            vf = np.asarray(vf, dtype=np.float32)

        # 3) GAE with the new rewards
        adv = np.zeros(T, dtype=np.float32)
        lastgaelam = 0.0
        for t in range(T - 1, -1, -1):
            nonterminal = 1.0 - float(dones[t])
            nextv = vf[t + 1] if t + 1 < T else 0.0
            delta = data["rewards"][t] + self.gamma * nextv * nonterminal - vf[t]
            lastgaelam = delta + self.gamma * self.lam * nonterminal * lastgaelam
            adv[t] = lastgaelam

        from ray.rllib.utils.sgd import standardized
        data["advantages"] = standardized(adv)
        data["value_targets"] = adv + vf

        return sb

    def _sync_workers_with_archive(self):
        """Sync rollout workers with a representative θ from the current archive.

        Choose the rep as:
          - best by fitness (if available), else
          - a random member (to avoid always using θ[0]).
        """
        if self.archive.population_size <= 0:
            return
        # Prefer fitness, else scalarized_value, else random
        scores = []
        for it in self.archive.archive:
            s = it.get("fitness", None)
            if s is None:
                s = it.get("scalarized_value", None)
            scores.append(s)
        if any(s is None for s in scores) or len(set(scores)) <= 1:
            rep_idx = int(np.random.randint(self.archive.population_size))
        else:
            rep_idx = int(np.argmax(scores))
        rep = self.archive.get_archive_item(rep_idx)
        if rep is None:
            return
        self.workers.local_worker().get_policy().set_weights(rep["theta"])
        self.workers.sync_weights()
        
    def _build_maml_split_on_episode_boundaries(self, batch, K: int):
        """Return a list[int] of length K that sums exactly to batch.count.
        We try to end each split at an episode boundary (DONE=True)."""
        # Total timesteps
        n = int(getattr(batch, "count", len(batch[SampleBatch.REWARDS])))

        # Try to get dones
        data = getattr(batch, "data", batch)
        dones = None
        if SampleBatch.DONES in data:
            dones = np.asarray(data[SampleBatch.DONES], dtype=np.bool_)
        else:
            term = data.get("terminateds", None)
            trunc = data.get("truncateds", None)
            if term is not None or trunc is not None:
                term = np.asarray(term, dtype=np.bool_) if term is not None else np.zeros(n, dtype=np.bool_)
                trunc = np.asarray(trunc, dtype=np.bool_) if trunc is not None else np.zeros(n, dtype=np.bool_)
                dones = np.asarray(term | trunc, dtype=np.bool_)

        # Start with near-equal arithmetic split
        base = n // K
        r = n - base * K
        sizes = [base] * K
        for j in range(r):
            sizes[j] += 1

        if dones is None:
            # No boundaries known -> ensure exact sum and return
            sizes[-1] = n - sum(sizes[:-1])
            return sizes

        # Push split ends to the next episode end for first K-1 splits
        out, offset = [], 0
        for j in range(K - 1):
            target = sizes[j]
            end = offset + target
            while end < n and not dones[end - 1]:
                end += 1
            # leave at least 1 step per remaining split
            end = min(end, n - (K - 1 - j))
            out.append(end - offset)
            offset = end
        out.append(n - sum(out))  # last split gets the rest

        # Safety: exact sum, all positive
        if sum(out) != n or len(out) != K or any(s <= 0 for s in out):
            out = sizes
            out[-1] = n - sum(out[:-1])
        return out
    
    def _gather_batch(self, rw, min_steps: int, need_eps: int = 1, cap: int = 16):
        gathered, steps, eps = [], 0, 0
        while (steps < min_steps or eps < need_eps) and cap > 0:
            mb = ray.get(rw.sample.remote())
            sb = convert_ma_batch_to_sample_batch(mb) if isinstance(mb, MultiAgentBatch) else mb
            gathered.append(sb); cap -= 1
            data = getattr(sb, "data", sb)
            n = int(getattr(sb, "count", len(data[SampleBatch.REWARDS])))
            steps += n
            d = data.get(SampleBatch.DONES, None)
            if d is not None:
                eps += int(np.asarray(d, dtype=np.bool_).sum())
            else:
                term = data.get("terminateds", None); trunc = data.get("truncateds", None)
                if term is not None or trunc is not None:
                    term = np.asarray(term, dtype=np.bool_) if term is not None else 0
                    trunc = np.asarray(trunc, dtype=np.bool_) if trunc is not None else 0
                    eps += int((term | trunc).sum())
                else:
                    seq = data.get("seq_lens", None)
                    eps += int(np.asarray(seq).size) if seq is not None else 1
        out = concat_samples(gathered) if len(gathered) > 1 else gathered[0]
        
        # === DRIVER-SIDE DEBUG: inspect env infos === ## debugging
        data = getattr(out, "data", out)

        # Safe fetch: don't use Python `or` with arrays
        infos = data.get(SampleBatch.INFOS, None)
        if infos is None:
            infos = data.get("infos", None)

        # Normalize to a python list of dicts if it's a numpy object array
        if isinstance(infos, np.ndarray) and infos.dtype == object:
            infos_list = infos.tolist()
        else:
            infos_list = infos

        if isinstance(infos_list, (list, tuple)) and len(infos_list) > 0:
            head = list(infos_list[:5])

            def _aslist(x):
                try:
                    return np.asarray(x).tolist()
                except Exception:
                    return x

            mr  = [(_aslist(i.get("mo_reward"))     if isinstance(i, dict) and "mo_reward"     in i else None)
                   for i in head]
            mrr = [(_aslist(i.get("mo_reward_raw")) if isinstance(i, dict) and "mo_reward_raw" in i else None)
                   for i in head]
        #     print("[DBG][_gather_batch] mo_reward first5:", mr)
        #     print("[DBG][_gather_batch] mo_reward_raw first5:", mrr)
        # else:
        #     print("[DBG][_gather_batch] no infos found; keys:", list(data.keys())) ## debugging
        
        return self._sanitize_samplebatch(out)  # CHANGE: sanitize sampler output
    
    def _gather_batch_local(self, lw, min_steps: int, need_eps: int = 1, cap: int = 16):
        gathered, steps, eps = [], 0, 0
        while (steps < min_steps or eps < need_eps) and cap > 0:
            sb = lw.sample()
            sb = convert_ma_batch_to_sample_batch(sb) if isinstance(sb, MultiAgentBatch) else sb
            gathered.append(sb); cap -= 1
            data = getattr(sb, "data", sb)
            n = int(getattr(sb, "count", len(data[SampleBatch.REWARDS])))
            steps += n
            d = data.get(SampleBatch.DONES, None)
            if d is not None:
                eps += int(np.asarray(d, dtype=np.bool_).sum())
            else:
                term = data.get("terminateds", None); trunc = data.get("truncateds", None)
                if term is not None or trunc is not None:
                    term = np.asarray(term, dtype=np.bool_) if term is not None else 0
                    trunc = np.asarray(trunc, dtype=np.bool_) if trunc is not None else 0
                    eps += int((term | trunc).sum())
                else:
                    seq = data.get("seq_lens", None)
                    eps += int(np.asarray(seq).size) if seq is not None else 1
        return concat_samples(gathered) if len(gathered) > 1 else gathered[0]
    

    def _equal_split(self, K: int, n: int) -> np.ndarray:
        base = max(1, n // K)
        s = [base] * K
        extra = n - base * K
        for j in range(K):
            if extra <= 0: break
            s[j] += 1; extra -= 1
        diff = n - sum(s)
        s[-1] += diff
        for j in range(K):
            if s[j] <= 0: s[j] = 1
        diff = n - sum(s)
        s[-1] += diff
        return np.asarray(s, dtype=np.int64)
    
    def _scalarize_and_make_advantages(self, batch, w, gamma: float, lam: float):
        sb = copy.deepcopy(batch)
        data = getattr(sb, "data", sb)
        if "mo_reward_vec" not in data:
            return sb
        R = np.asarray(data["mo_reward_vec"], dtype=np.float32)
        r = R @ w[: R.shape[1]]
        data[SampleBatch.REWARDS] = r.astype(np.float32)

        T = len(r)
        dones = data.get(SampleBatch.DONES, None)
        if dones is not None:
            dones = np.asarray(dones, dtype=np.bool_)
        else:
            term = data.get("terminateds", None); trunc = data.get("truncateds", None)
            if term is not None or trunc is not None:
                term = np.asarray(term, dtype=np.bool_) if term is not None else np.zeros(T, dtype=np.bool_)
                trunc = np.asarray(trunc, dtype=np.bool_) if trunc is not None else np.zeros(T, dtype=np.bool_)
                dones = np.asarray(term | trunc, dtype=np.bool_)
            else:
                seq = data.get("seq_lens", None)
                if seq is not None:
                    seq = np.asarray(seq, dtype=np.int32)
                    dones = np.zeros(T, dtype=np.bool_)
                    ends = np.cumsum(seq) - 1
                    ends = ends[(ends >= 0) & (ends < T)]
                    dones[ends] = True
                else:
                    dones = np.zeros(T, dtype=np.bool_); dones[-1] = True

        Rt = np.zeros(T, dtype=np.float32)
        running = 0.0
        for t in range(T - 1, -1, -1):
            if dones[t]:
                running = 0.0
            running = r[t] + gamma * running
            Rt[t] = running

        data["advantages"] = standardized(Rt)
        data["value_targets"] = Rt.astype(np.float32)
        return sb

    def _ensure_maml_split(self, batch: SampleBatch, K: int, M: int = 1) -> SampleBatch:
        """Force batch['split'] to shape (M, K) with positive ints that sum to batch.count."""
        n = int(getattr(batch, "count", len(batch[SampleBatch.REWARDS])))
        want_shape = (M, K)

        sp = batch.get("split", None)
        if sp is None:
            row = self._equal_split(K, n)                 # (K,)
            sp = row.reshape(1, K)                        # (1, K)
        else:
            sp = np.asarray(sp, dtype=np.int64)
            # Accept common user shapes and coerce to (M, K)
            if sp.ndim == 1:                              # (K,)
                sp = sp.reshape(1, -1)
            elif sp.ndim == 2 and sp.shape == (K, 1):     # (K,1) -> (1,K)
                sp = sp.T
            elif sp.ndim != 2:
                sp = sp.reshape(1, -1)

            # If still wrong dims, try transpose once.
            if sp.shape != want_shape:
                if sp.T.shape == want_shape:
                    sp = sp.T
                else:
                    # Rebuild a fresh, valid split
                    row = self._equal_split(K, n)
                    sp = row.reshape(1, K)

        # Strict positivity + exact sum per task (M=1 here)
        if np.any(sp <= 0) or int(sp.sum()) != n * M:
            row = self._equal_split(K, n)
            sp = row.reshape(1, K)

        batch["split"] = sp.astype(np.int64)
        return batch
    
    def _sanitize_samplebatch(self, sb, clip_val: float = 1e6):
        """Make sure everything the loss might touch is finite."""
        data = getattr(sb, "data", sb)

        def _fix_arr(x):
            arr = np.asarray(x)
            if arr.dtype.kind in "f":  # floats only
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                if np.isfinite(clip_val):
                    arr = np.clip(arr, -clip_val, clip_val)
                return arr.astype(np.float32, copy=False)
            return arr

        # Handle obs (can be array or dict for dict-obs spaces)
        if SampleBatch.OBS in data:
            obs = data[SampleBatch.OBS]
            if isinstance(obs, dict):
                fixed = {}
                for k, v in obs.items():
                    fixed[k] = _fix_arr(v)
                data[SampleBatch.OBS] = fixed
            else:
                data[SampleBatch.OBS] = _fix_arr(obs)

        # Some stacks also store new_obs/next_obs/obs_flat
        for k in (SampleBatch.NEXT_OBS, "new_obs", "obs_flat"):
            if k in data:
                v = data[k]
                if isinstance(v, dict):
                    fixed = {kk: _fix_arr(vv) for kk, vv in v.items()}
                    data[k] = fixed
                else:
                    data[k] = _fix_arr(v)

        # Actions can be float for continuous envs
        if SampleBatch.ACTIONS in data:
            data[SampleBatch.ACTIONS] = _fix_arr(data[SampleBatch.ACTIONS])

        # Rewards/advantages/value_targets and your MO vectors
        D = int(getattr(self.archive, "num_objectives", 0) or 0)
        if (not D) and ("mo_reward_vec" in data and isinstance(data["mo_reward_vec"], np.ndarray) and data["mo_reward_vec"].ndim == 2):
            D = int(data["mo_reward_vec"].shape[1])

        dynamic_reward_keys = [f"rewards-{d}" for d in range(max(D, 0))]

        for k in (
            SampleBatch.REWARDS,
            "advantages",
            "value_targets",
            "mo_reward_vec",         # keep sanitizing the packed matrix
            *dynamic_reward_keys,    # <-- replaces mo_profit/emiss/ineq
        ):
            if k in data:
                data[k] = _fix_arr(data[k])
                
        return sb
# ==========================
# Training utilities
# ==========================

def post_process_metrics(adapt_iter, workers, metrics):
    name = "_adapt_" + str(adapt_iter) if adapt_iter > 0 else ""
    res = collect_metrics(workers=workers)
    metrics["episode_reward_max" + str(name)] = res["episode_reward_max"]
    metrics["episode_reward_mean" + str(name)] = res["episode_reward_mean"]
    metrics["episode_reward_min" + str(name)] = res["episode_reward_min"]
    return metrics


def inner_adaptation(workers, samples):
    """Each worker performs one gradient descent step on its shard."""
    for i, e in enumerate(workers.remote_workers()):
        sample_batch = samples[i] if isinstance(samples, list) else samples
        e.learn_on_batch.remote(sample_batch)

# ==========================
# Algorithm
# ==========================

class MERLION(Algorithm):
    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return MERLIONConfig()

    @classmethod
    @override(Algorithm)
    def get_default_policy_class(
        cls, config: AlgorithmConfig
    ) -> Optional[Type[Policy]]:
        if config["framework"] == "torch":
            from ray.rllib.algorithms.maml.maml_torch_policy import MAMLTorchPolicy
            return MAMLTorchPolicy
        elif config["framework"] == "tf":
            from ray.rllib.algorithms.maml.maml_tf_policy import MAMLTF1Policy
            return MAMLTF1Policy
        else:
            from ray.rllib.algorithms.maml.maml_tf_policy import MAMLTF2Policy
            return MAMLTF2Policy

    @staticmethod
    @override(Algorithm)

    def execution_plan(workers: WorkerSet, config: AlgorithmConfig, **_kwargs) -> LocalIterator[dict]:
        assert len(_kwargs) == 0, "MERLION execution_plan does NOT take additional parameters"

        # Torch-only guard because the provided generators instantiate torch policies
        assert config["framework"] == "torch", "Initializers assume PyTorch policies."

        # ========= Build diverse initial thetas (unchanged) =========
        base_policy = workers.local_worker().get_policy()
        policy_cls  = type(base_policy)
        obs_space   = base_policy.observation_space
        act_space   = base_policy.action_space
        pol_cfg     = copy.deepcopy(base_policy.config)
        pol_cfg["meta_batch_size"] = 1

        base_w = base_policy.get_weights()
        
        method = getattr(config, "init_method", "mixed")
        init_kwargs = getattr(config, "init_kwargs", {}) or {}

        if method in ("seed", "scale", "mixed", "pretrained"):
            initial_thetas = generate_diverse_meta_policies(
                policy_cls, obs_space, act_space, pol_cfg,
                population_size=config.population_size,
                diversity_method=method, **init_kwargs
            )
        elif method == "orthogonal":
            initial_thetas = generate_orthogonal_meta_policies(
                policy_cls, obs_space, act_space, pol_cfg,
                population_size=config.population_size
            )
        elif method == "strategy":
            initial_thetas = generate_strategy_based_meta_policies(
                policy_cls, obs_space, act_space, pol_cfg,
                population_size=config.population_size
            )
        else:
            initial_thetas = [copy.deepcopy(base_w) for _ in range(config.population_size)]

            
        def _sanitize_theta(theta, ref):
            cleaned = {}
            for k, v in theta.items():
                arr = np.asarray(v)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                arr = np.clip(arr, -1e3, 1e3)
                cleaned[k] = arr.astype(v.dtype, copy=False)
            for k, v in cleaned.items():
                if not np.all(np.isfinite(v)):
                    return {kk: np.asarray(ref[kk]).copy() for kk in ref}
            return cleaned

        initial_thetas = [_sanitize_theta(th, base_w) for th in initial_thetas]
    
        # Archive constructed with the LIST of θ’s (unchanged)
        archive = MERLIONArchive(config.population_size, config.num_objectives, initial_thetas)  # :contentReference[oaicite:1]{index=1}
        workers._merlion_archive = archive ## additional for deserialisation

        # ========= NEW: seed each remote worker with its own θ_i =========
        remotes = workers.remote_workers()
        P = int(config.population_size)
        worker_for_i = []

        if len(remotes) == 0:
            logger.warning("No remote workers detected; falling back to local-only (reduced diversity parallelism).")
            # Fallback mapping to local worker (sampling will still work but not in parallel)
            worker_for_i = [workers.local_worker() for _ in range(P)]
        else:
            for i in range(P):
                rw = remotes[i % len(remotes)]
                theta_i = archive.get_archive_item(i)["theta"]  # or: initial_thetas[i]
                # Set θ_i on that remote worker ONLY
                rw.foreach_policy.remote(lambda p, pid, th=theta_i: p.set_weights(th))
                worker_for_i.append(rw)

        # IMPORTANT: do NOT broadcast a representative θ to all workers.
        # Your previous code did:
        #   workers.local_worker().get_policy().set_weights(initial_thetas[0])
        #   workers.sync_weights()
        # which homogenised all workers. We remove that. :contentReference[oaicite:2]{index=2}
        #
        # It's fine to set the LOCAL policy to something for initial bookkeeping;
        # it won't be synced to remotes:
        workers.local_worker().get_policy().set_weights(initial_thetas[0])

        # Keep your meta-env task refresh
        use_meta_env = config.use_meta_env
        set_worker_tasks(workers, use_meta_env)

        # Metrics collector (unchanged)
        metric_collect = CollectMetrics(
            workers,
            min_history=config.metrics_num_episodes_for_smoothing,
            timeout_seconds=config.metrics_episode_collection_timeout_s,
        )

        inner_steps = int(config.inner_adaptation_steps)

        # ========= Iterator plumbing (kept compatible) =========
        def inner_adaptation_steps(itr):
            """
            Batches shards and emits (out, metrics) tuples.
            We keep this so metrics flow is unchanged, but the actual training
            will use per-i sampling inside MERLIONMetaUpdate.
            """
            buf = []
            split = []
            metrics = {}
            for samples in itr:
                split_lst = []
                for sample in samples:
                    sample = convert_ma_batch_to_sample_batch(sample)
                    # Optional advantage standardization BEFORE meta (gate it to avoid double-normalizing later)
                    if getattr(config, "standardize_before_meta", False):
                        sample["advantages"] = standardized(sample["advantages"])
                    split_lst.append(sample.count)
                    buf.append(sample)
                split.append(split_lst)

                adapt_iter = len(split) - 1
                metrics = post_process_metrics(adapt_iter, workers, metrics)

                if len(split) > inner_steps:
                    out = concat_samples(buf)
                    out["split"] = np.array(split)
                    buf = []
                    split = []

                    ep_rew_pre  = metrics["episode_reward_mean"]
                    ep_rew_post = metrics["episode_reward_mean_adapt_" + str(inner_steps)]
                    metrics["adaptation_delta"] = ep_rew_post - ep_rew_pre

                    yield out, metrics
                    metrics = {}
                else:
                    inner_adaptation(workers, samples)

        rollouts = from_actors(workers.remote_workers())
        rollouts = rollouts.batch_across_shards()
        rollouts = rollouts.transform(inner_adaptation_steps)

        # ========= Pass worker mapping into MERLIONMetaUpdate =========
        train_op = rollouts.for_each(
            MERLIONMetaUpdate(
                workers=workers,
                maml_steps=config.maml_optimizer_steps,
                metric_gen=metric_collect,
                use_meta_env=use_meta_env,
                archive=archive,
                worker_for_i=worker_for_i,   # << NEW: per-i sampling mapping
            )
        )
        return train_op


# Deprecated: Use MERLIONConfig instead
class _deprecated_default_config(dict):
    def __init__(self):
        super().__init__(MERLIONConfig().to_dict())

    @Deprecated(
        old="ray.rllib.algorithms.merlion.merlion.DEFAULT_CONFIG",
        new="ray.rllib.algorithms.merlion.merlion.MERLIONConfig(...)",
        error=True,
    )
    def __getitem__(self, item):
        return super().__getitem__(item)


DEFAULT_CONFIG = _deprecated_default_config()
