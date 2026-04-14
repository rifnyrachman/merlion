# pip install mo-gymnasium gymnasium[mujoco] ray[rllib] torch mujoco

from typing import Dict, List, Optional, Sequence, Union
from types import SimpleNamespace

import numpy as np
import gymnasium as gym
import mo_gymnasium as mo_gym
from gymnasium.spaces import Box, Discrete

def _dirichlet_weight(d: int) -> np.ndarray:
    w = np.random.dirichlet(np.ones(d))
    return w.astype(np.float32)


def _infer_reward_dim(env: gym.Env, fallback: Optional[int] = None) -> int:
    # Try reward_space (MO-Gymnasium convention)
    rs = getattr(env.unwrapped, "reward_space", None)
    if rs is not None and hasattr(rs, "shape") and len(rs.shape) == 1:
        return int(rs.shape[0])
    # Try sample step without advancing training state (only if safe)
    try:
        _obs, _info = env.reset(seed=None)
        a = env.action_space.sample()
        _obs2, r_vec, _t, _tr, _inf = env.step(a)
        if isinstance(r_vec, (list, tuple, np.ndarray)):
            d = len(r_vec)
            # Try to rewind if possible
            try:
                env.reset(seed=None)
            except Exception:
                pass
            return int(d)
    except Exception:
        pass
    if fallback is not None:
        return int(fallback)
    raise RuntimeError("Could not infer reward dimension; please set reward_dim in config.")


def _default_sign_vector(env_id: str, d: int) -> np.ndarray:
    # +1 means “maximize as-is”, -1 means “this is a positive cost -> flip sign to treat as a benefit”
    # Defaults chosen from MO-Gymnasium docs/conventions; code will pad/truncate to match d.
    defaults = {
        # Grid / discrete classics
        "deep-sea-treasure-v0":         [1, 1],          # [treasure (+), time penalty (−)] → maximize both as-is
        "resource-gathering-v0":        [1, 1, 1],       # [death penalty (−), gold (+), diamond (+)] → maximize as-is
        "four-room-v0":                 [1, 1, 1],       # three collectible rewards are positive events

        # MountainCar
        "mo-mountaincar-v0":            [1, 1],          # progress (+), energy/time penalty typically (−) → keep as-is
        "mo-mountaincarcontinuous-v0":  [1, 1],          # ditto for continuous control

        # LunarLander (Box2D)
        # 4D: [crash/land outcome, shaping, fuel main, fuel side]; fuel costs are negative → keep as-is
        "mo-lunar-lander-v2":           [1, 1, 1, 1],

        # Water-Reservoir (continuous)
        # up to 4 “cost/deficit” components typically ≤ 0 by design → keep as-is
        "water-reservoir-v0":           [1, 1, 1, 1],

        # Highway
        # 3D: [speed reward (+), right-lane reward (+), collision reward (≤0)] → keep as-is
        "mo-highway-fast-v0":           [1, 1, 1],

        # MuJoCo MO variants
        # Common convention: progress/velocity (+) and control cost (−) → keep as-is
        "mo-halfcheetah-v4":            [1, 1],
        "mo-hopper-v4":                 [1, 1],
        # MO-Reacher returns 4 target-aligned terms r_i = 1 − 4||dist_i||^2 (higher is better) → keep as-is
        "mo-reacher-v4":                [1, 1, 1, 1],
    }

    vec = defaults.get(env_id, [1] * d)
    if len(vec) != d:
        if len(vec) < d:
            vec = list(vec) + [1] * (d - len(vec))  # pad with +1 (maximize as-is)
        else:
            vec = vec[:d]  # truncate if env configured with fewer objectives
    return np.asarray(vec, dtype=np.float32)

class ScalarizationWeightProxy_notrandom(gym.Wrapper):
    def set_scalarization_weights(self, w):
        if hasattr(self.env, "set_scalarization_weights"):
            return self.env.set_scalarization_weights(w)
        raise AttributeError("Underlying env has no set_scalarization_weights(w)")

    def get_scalarization_weights(self):
        if hasattr(self.env, "get_scalarization_weights"):
            return self.env.get_scalarization_weights()
        return None

class ScalarizedMetaEnv_notrandom(gym.Env):
    """
    Multi-objective scalarization wrapper with optional task perturbations
    and reward normalization. Works across Mujoco and non-Mujoco MO-Gymnasium
    environments by inferring reward dimensionality and using a configurable
    orientation (sign_vector) to make all objectives “higher is better”.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Dict):
        # ----- Base config
        self.base_id: str = config.get("base_id", "mo-halfcheetah-v4")
        # if self.base_id not in ['deep-sea-treasure-v0', 'resource-gathering-v0', 'mo-mountaincar-v0', 'mo-mountaincarcontinuous-v0', 'mo-lunar-lander-v2']:
        self.max_episode_steps = int(config.get("max_episode_steps", 1000))
    
        self.env = mo_gym.make(self.base_id)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.d_obj: int = int(config.get("reward_dim", _infer_reward_dim(self.env, fallback=2)))

        # Scalarization weight
        w = np.asarray(config.get("weight", None), dtype=np.float32) if config.get("weight", None) is not None else _dirichlet_weight(self.d_obj)
        self._w: np.ndarray = self._normalize_weight(w)

        # ----- Orientation and scaling
        # sign_vector: +1 (maximize as-is), -1 (minimize/cost -> flipped)
        sign_vec_cfg = config.get("sign_vector", None)
        if sign_vec_cfg is None:
            self.sign_vector: np.ndarray = _default_sign_vector(self.base_id, self.d_obj)
        else:
            self.sign_vector = np.asarray(sign_vec_cfg, dtype=np.float32)
            if self.sign_vector.shape[0] != self.d_obj:
                raise ValueError("sign_vector length must match reward_dim")

        # Optional per-objective scaling after orientation (e.g., ctrl cost scale)
        scales = config.get("obj_scales", None)
        if scales is None:
            self.obj_scales: np.ndarray = np.ones(self.d_obj, dtype=np.float32)
        else:
            self.obj_scales = np.asarray(scales, dtype=np.float32)
            if self.obj_scales.shape[0] != self.d_obj:
                raise ValueError("obj_scales length must match reward_dim")

        # ----- Normalization (EMA)
        # self.normalize_objectives: bool = bool(config.get("normalize_objectives", True))
        self.normalize_objectives: bool = False
        self.norm_scope: str = str(config.get("norm_scope", "task"))  # "task" or "global"
        self.norm_mode: str = str(config.get("norm_mode", "ema"))
        self.norm_beta: float = float(config.get("norm_beta", 0.01))
        self.norm_warmup_steps: int = int(config.get("norm_warmup_steps", 100))
        self.norm_freeze: bool = bool(config.get("norm_freeze", False))

        self._task_norm: Dict[int, Dict[str, np.ndarray]] = {}
        self._obj_mean: Optional[np.ndarray] = None
        self._obj_var: Optional[np.ndarray] = None

        # Cache original Mujoco physics (if present)
        un = self.env.unwrapped
        self._orig_body_mass = un.model.body_mass.copy() if hasattr(un, "model") else None
        self._orig_geom_friction = un.model.geom_friction.copy() if hasattr(un, "model") else None

        # Current task and bookkeeping
        self.task = self._sample_task_obj()
        self.task_id: Optional[int] = None
        self.task_weights: Dict[int, List[np.ndarray]] = {}

        self._apply_task(self._task_to_dict(self.task))
        self.t = 0

        # Episode bookkeeping
        self.timestep = 0

        # Build base env dari MO-Gymnasium
        self.max_timestep: int = int(config.get("max_episode_steps", 1000))
        
        # Reward bookkeeping (shape sesuai num_objectives)
        self.vector_reward = np.zeros(self.d_obj, dtype=np.float32)
        self.scalar_reward = 0.0

        # Velocity fallback (untuk MuJoCo envs)
        self._last_x = None
        self._last_obs = None

        # Task state (no-arg set_task() -> schedule a random seed for next reset)
        self._current_task = {"env_id": self.base_id, "seed": None}
        self._pending_seed = None  # applied on next reset()

    # -------- scalarization weights API (same names) --------
    def set_scalarization_weights(self, w):
        w = np.asarray(w, dtype=np.float32).reshape(-1)
        if w.size != self.d_obj:
            raise ValueError(f"Expected {self.d_obj} weights, got {w.size}")
        w = np.clip(w, 1e-12, None)
        self._w = (w / w.sum()).astype(np.float32)

    def get_scalarization_weights(self):
        return None if self._w is None else self._w.copy()

    # -------- tasks API (aligned to your attached file) --------
    def sample_tasks(self, n_tasks):
        seeds = np.random.randint(0, 2**31 - 1, size=n_tasks, dtype=np.int64)
        return [{"env_id": self.base_id, "seed": int(s)} for s in seeds]

    def set_task(self):
        """
        No-arg set_task() per the attached structure.
        IMPORTANT: Do NOT recreate or reset base_env here.
        We simply schedule a new random seed to be applied on the next reset().
        """
        self._pending_seed = int(np.random.randint(0, 2**31 - 1))

    def get_task(self):
        return dict(self._current_task)

    def set_weight(self, w: Union[List[float], np.ndarray]):
        self._w = self._normalize_weight(np.asarray(w, dtype=np.float32))

    # Bulk weight setter for MIRACL: {task_id: [w_k, ...]}
    def set_weights(self, weights_dict: Dict[int, List[Union[List[float], np.ndarray]]]):
        
        if isinstance(weights_dict, np.ndarray):
            # Case 1a: 2D array → multiple weights for single task
            if weights_dict.ndim == 2:
                self.task_weights[self.task_id] = [
                    self._normalize_weight(np.asarray(w, dtype=np.float32)) for w in weights_dict
                ]
            # Case 1b: 1D array → single weight vector
            elif weights_dict.ndim == 1:
                self.task_weights[self.task_id] = [
                    self._normalize_weight(np.asarray(weights_dict, dtype=np.float32))
                ]
        elif isinstance(weights_dict, dict):
            for tid, w_list in weights_dict.items():
                self.task_weights[tid] = [self._normalize_weight(np.asarray(w, dtype=np.float32)) for w in w_list]

    # ========= Gym API =========
    def reset(self, *, seed=None, options=None):
        self.t = 0
        
        if self.task_id is None:
            self.task_id = -1
        if self.norm_scope == "task":
            self._ensure_task_norm(self.task_id, self.d_obj)
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        # Apply action noise only for continuous actions
        if isinstance(self.action_space, gym.spaces.Box):
            act_noise_std = float(getattr(self, "act_noise_std", 0.0))
            if act_noise_std > 0:
                noise = np.random.normal(0, act_noise_std, size=np.shape(action))
                action = action + noise
                # Clip to action bounds
                if self.action_space.is_bounded("both"):
                    action = np.clip(action, self.action_space.low, self.action_space.high)
                    
        obs, r_vec_raw, term, trunc, info = self.env.step(action)
        
        if self.observation_space.is_bounded("both"):
            obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        
        r_vec_raw = np.asarray(r_vec_raw, dtype=np.float32)

        # Orient -> scale -> optional normalize
        r_vec_oriented = self.sign_vector * r_vec_raw
        r_vec_oriented = r_vec_oriented * self.obj_scales

        if self.normalize_objectives and self.norm_mode == "ema":
            if self.norm_scope == "task":
                tid = self.task_id if self.task_id is not None else -1
                self._ensure_task_norm(tid, len(r_vec_oriented))
                self._update_task_norm(tid, r_vec_oriented)
                st = self._task_norm[tid]
                if st["steps"] >= self.norm_warmup_steps:
                    r_vec_used = self._normalize_task(tid, r_vec_oriented)
                else:
                    r_vec_used = r_vec_oriented
            else:
                self._maybe_init_norm(r_vec_oriented)
                self._update_norm(r_vec_oriented)
                r_vec_used = self._normalize(r_vec_oriented)
        else:
            r_vec_used = r_vec_oriented

        # Scalarize; guard against accidental mismatch
        if self._w.shape[0] != r_vec_used.shape[0]:
            raise ValueError(f"Weight dim {self._w.shape[0]} != reward dim {r_vec_used.shape[0]}")
        r_scalar = float(np.dot(self._w, r_vec_used))

        self.vector_reward = r_vec_used.copy()
        
        # Scalarize
        if self._w is None:
            self.set_scalarization_weights(np.ones(self.d_obj, dtype=np.float32))
        self.scalar_reward = float(np.dot(self.vector_reward, self._w))

        # Info
        info = dict(info)
        info["r_vec_raw"] = r_vec_raw
        info["r_vec_oriented"] = r_vec_oriented
        info["r_vec_used"] = r_vec_used
        info["w"] = self._w
        info["task_id"] = self.task_id
        if self.norm_scope == "task":
            st = self._task_norm[self.task_id if self.task_id is not None else -1]
            info["norm_steps"] = st["steps"]
            info["norm_mean"] = st["mean"]
            info["norm_var"] = st["var"]
            info["norm_frozen"] = st["frozen"]
        else:
            if self._obj_mean is not None:
                info["norm_mean"] = self._obj_mean
                info["norm_var"] = self._obj_var

        if self.max_episode_steps is not None:
            self.t += 1
            if self.t >= self.max_episode_steps:
                trunc = True

        info["mo_reward"] = self.vector_reward.copy()
        info["mo_reward_raw"] = r_vec_raw.copy()
        info["weights"] = self._w.copy()

        return obs, r_scalar, term, trunc, info

    # ========= Internals =========
    @staticmethod
    def _normalize_weight(w: np.ndarray) -> np.ndarray:
        s = float(w.sum())
        if s <= 0:
            raise ValueError("Weight vector must be positive and sum > 0.")
        return (w / s).astype(np.float32)

    def _sample_task_obj(self):
        t = {}
        # Placeholder vector (length d_obj) for downstream metadata usage
        t["vector_reward"] = np.zeros(self.d_obj, dtype=np.float32)
        return SimpleNamespace(**t)

    def _task_to_dict(self, task) -> Dict[str, float]:
        if isinstance(task, dict):
            return task
        # keys = ["mass_scale", "friction_scale", "ctrl_cost_scale", "act_noise_std"]
        # return {k: getattr(task, k) for k in keys if hasattr(task, k)}

    def _apply_task(self, task: dict):
        """
        Apply parameters in `task` dict to self.env.unwrapped and self
        in a general way (no Mujoco special handling).
        """
        un = self.env.unwrapped
        if task is not None:
            for k, v in task.items():

                if hasattr(un, k):
                    setattr(un, k, v)

                elif hasattr(self, k):
                    setattr(self, k, v)

    # ---- Global EMA stats
    def _maybe_init_norm(self, r_vec: np.ndarray):
        if self._obj_mean is None:
            d = len(r_vec)
            self._obj_mean = np.zeros(d, dtype=np.float32)
            self._obj_var = np.ones(d, dtype=np.float32)

    def _update_norm(self, r_vec: np.ndarray):
        beta = self.norm_beta
        delta = r_vec - self._obj_mean
        self._obj_mean += beta * delta
        self._obj_var += beta * (delta ** 2 - self._obj_var)

    def _normalize(self, r_vec: np.ndarray) -> np.ndarray:
        eps = 1e-8
        return (r_vec - self._obj_mean) / np.sqrt(self._obj_var + eps)

    # ---- Per-task EMA stats
    def _ensure_task_norm(self, task_id: int, d: int):
        if task_id not in self._task_norm:
            self._task_norm[task_id] = {
                "mean": np.zeros(d, dtype=np.float32),
                "var": np.ones(d, dtype=np.float32),
                "steps": 0,
                "frozen": False,
            }

    def _update_task_norm(self, task_id: int, r_vec: np.ndarray):
        st = self._task_norm[task_id]
        if st["frozen"]:
            return
        beta = self.norm_beta
        delta = r_vec - st["mean"]
        st["mean"] += beta * delta
        st["var"] += beta * (delta ** 2 - st["var"])
        st["steps"] += 1
        if self.norm_freeze and st["steps"] >= self.norm_warmup_steps:
            st["frozen"] = True

    def _normalize_task(self, task_id: int, r_vec: np.ndarray) -> np.ndarray:
        st = self._task_norm[task_id]
        eps = 1e-8
        return (r_vec - st["mean"]) / np.sqrt(st["var"] + eps)

def make_env_not_random(env_config=None):
    """
    Factory function untuk RLlib.
    env_config dapat berisi {"env_id": "mo-halfcheetah-v4", "num_objectives": 2}
    """
    if env_config is None:
        env_config = {}
    
    env_id = env_config.get("env_id", "mo-halfcheetah-v4")
    num_objectives = env_config.get("num_objectives", None)
    
    base = ScalarizedMetaEnv_notrandom(env_id=env_id, num_objectives=num_objectives)
    return ScalarizationWeightProxy_notrandom(base)
