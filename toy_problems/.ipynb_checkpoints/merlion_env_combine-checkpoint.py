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

class ScalarizationWeightProxy(gym.Wrapper):
    def set_scalarization_weights(self, w):
        if hasattr(self.env, "set_scalarization_weights"):
            return self.env.set_scalarization_weights(w)
        raise AttributeError("Underlying env has no set_scalarization_weights(w)")

    def get_scalarization_weights(self):
        if hasattr(self.env, "get_scalarization_weights"):
            return self.env.get_scalarization_weights()
        return None

class ScalarizedMetaEnv(gym.Env):
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
        # print(f'task: {self.task}')

        self._apply_task(self._task_to_dict(self.task))
        self.t = 0
        self.timestep = 0
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


    def sample_tasks(self, n_tasks: int):
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

    # ========= Gym API =========
    def reset(self, *, seed=None, options=None):

        if seed is not None:
            use_seed = seed
        elif self._pending_seed is not None:
            use_seed = self._pending_seed
        else:
            use_seed = None
        
        self.t = 0
        self.env = reset_env1(self.env, self.base_id)
        
        if self.task_id is None:
            self.task_id = -1
        if self.norm_scope == "task":
            self._ensure_task_norm(self.task_id, self.d_obj)
        obs, info = self.env.reset(seed=seed, options=options)
        
        self._current_task = {"env_id": self.base_id, "seed": use_seed}
        self._pending_seed = None  # consumed

        self._last_obs = obs
        self._last_x = info.get("x_position", None) if isinstance(info, dict) else None
        
        if self._w is None:
            self.set_scalarization_weights(np.ones(self.d_obj, dtype=np.float32))

        self.timestep = 0
        self.vector_reward[:] = 0.0
        self.scalar_reward = 0.0
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
        t = randomize_parameter_env(self.env, self.base_id)
        # Placeholder vector (length d_obj) for downstream metadata usage
        t["vector_reward"] = np.zeros(self.d_obj, dtype=np.float32)
        return SimpleNamespace(**t)

    def _task_to_dict(self, task) -> Dict[str, float]:
        if isinstance(task, dict):
            return task

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

def make_env(env_config=None):
    """
    Factory function untuk RLlib.
    env_config dapat berisi {"env_id": "mo-halfcheetah-v4", "num_objectives": 2}
    """
    if env_config is None:
        env_config = {}
    
    env_id = env_config.get("env_id", "mo-halfcheetah-v4")
    num_objectives = env_config.get("num_objectives", None)
    
    base = ScalarizedMetaEnv(env_id=env_id, num_objectives=num_objectives)
    return ScalarizationWeightProxy(base)

def randomize_parameter_env(env, env_name):
    params = get_parameter_env(env.unwrapped, env_name)

    rnd = {}
    rng = np.random.uniform

    if env_name == 'mo-halfcheetah-v4':
        rnd['_forward_reward_weight'] = params['_forward_reward_weight'] * rng(0.5, 1.5)
        rnd['_ctrl_cost_weight']      = params['_ctrl_cost_weight']      * rng(0.5, 1.5)
        rnd['_reset_noise_scale']     = params['_reset_noise_scale']     * rng(0.5, 2.0)

    elif env_name == 'mo-hopper-v4':
        rnd["_forward_reward_weight"]    = params["_forward_reward_weight"] * rng(0.5, 1.5)
        rnd["_ctrl_cost_weight"]         = params["_ctrl_cost_weight"]      * rng(0.5, 1.5)
        rnd["_healthy_reward"]           = params["_healthy_reward"]        * rng(0.5, 1.5)

        rnd["_terminate_when_unhealthy"] = bool(np.random.rand() < 0.5)
        rnd["_healthy_state_range"]      = (params["_healthy_state_range"][0]   * rng(0.8, 1.2), params["_healthy_state_range"][1]   * rng(0.8, 1.2))
        # rnd["_healthy_z_range"][0]       = params["_healthy_z_range"][0]       * rng(0.5, 1.5)
        rnd["_healthy_angle_range"]      = (params["_healthy_angle_range"][0]   * rng(0.5, 1.5), params["_healthy_angle_range"][1]   * rng(0.5, 1.5))
        rnd["_reset_noise_scale"]        = params["_reset_noise_scale"]     * rng(0.5, 2.0)

    elif env_name == 'mo-mountaincar-v0':
        rnd['force']   = params['force']   * rng(0.5, 1.5)
        rnd['gravity'] = params['gravity'] * rng(0.5, 1.5)

    elif env_name == 'mo-mountaincarcontinuous-v0':
        rnd['power'] = params['power'] * rng(0.5, 1.5)

    elif env_name == 'mo-lunar-lander-v2':

        g = params['gravity'] * rng(0.5, 1.5)
        g = float(np.clip(g, -12.0, -1e-3))
        w = params['wind_power'] * rng(0.5, 1.5)
        w = float(np.clip(w, 0.0, 20.0))
        t = params['turbulence_power'] * rng(0.5, 1.5)
        t = float(np.clip(t, 0.0, 2.0))
        rnd['gravity']          = g
        rnd['wind_power']       = w
        rnd['turbulence_power'] = t

    elif env_name == 'water-reservoir-v0':
        rnd['nO']        = int(round(params['nO'] * rng(0.5, 1.5)))
        rnd['penalize']  = bool(np.random.rand() < 0.5)
        rnd['time_limit'] = int(round(params['time_limit'] * rng(0.5, 1.5)))
    elif env_name == 'resource-gathering-v0':
        size = np.random.choice([5, 7, 9], 1)[0]
        nr = 2 if size == 5 else np.random.choice([3, 4, 5, 6], 1)[0]

        ne = 2 if size == 5 else np.random.choice([3, 4], 1)[0]
        map1, init_pos = sample_resource_gathering_map(
            s = size, 
            n_R = nr, 
            n_E = ne
        )
        rnd['map'] = map1
        rnd['initial_pos'] = init_pos
    elif env_name == 'four-room-v0':
        rnd['maze'] = sample_four_room_maze(
            n1=np.random.randint(3,5), n2=np.random.randint(3,5), 
            n3=np.random.randint(3,5), seed=np.random.randint(0, 10000)
        )
    elif env_name == 'deep-sea-treasure-v0':
        sm, pf = sample_treasures_from_map(noise_scale=0.5, 
                                                   seed=np.random.randint(0, 10000))
        rnd['sea_map'] = sm
        rnd['_pareto_front'] = pf

    elif env_name == 'mo-highway-fast-v0':
        rnd = {
        'lanes_count': np.random.randint(2,5), 
        'vehicles_count': int(20 * np.random.uniform(0.5, 1.5, 1)), 
        'controlled_vehicles': np.random.randint(1,3), 
        'ego_spacing': 1.5 * np.random.uniform(0.5, 1.5, 1)
        }
    return rnd

def reset_env1(env, env_name):
    params = get_parameter_env(env.unwrapped, env_name)
    rng = np.random.uniform
    env1 = mo_gym.make(env_name)

    if env_name == 'mo-halfcheetah-v4':
        env1._forward_reward_weight = params['_forward_reward_weight'] * rng(0.5, 1.5)
        env1._ctrl_cost_weight      = params['_ctrl_cost_weight']      * rng(0.5, 1.5)
        env1._reset_noise_scale     = params['_reset_noise_scale']     * rng(0.5, 2.0)

    elif env_name == 'mo-hopper-v4':
        env1._forward_reward_weight    = params["_forward_reward_weight"] * rng(0.5, 1.5)
        env1._ctrl_cost_weight         = params["_ctrl_cost_weight"]      * rng(0.5, 1.5)
        env1._healthy_reward           = params["_healthy_reward"]        * rng(0.5, 1.5)
        env1._terminate_when_unhealthy = bool(np.random.rand() < 0.5)
        env1._healthy_state_range      = (params["_healthy_state_range"][0]   * rng(0.8, 1.2), params["_healthy_state_range"][1]   * rng(0.8, 1.2))
        # env1._healthy_z_range[0]       = params["_healthy_z_range"][0]       * rng(0.5, 1.5)
        env1._healthy_angle_range      = (params["_healthy_angle_range"][0]   * rng(0.5, 1.5), params["_healthy_angle_range"][1]   * rng(0.5, 1.5))
        env1._reset_noise_scale        = params["_reset_noise_scale"]     * rng(0.5, 2.0)

    elif env_name == 'mo-mountaincar-v0':
        env1.force   = params['force']   * rng(0.5, 1.5)
        env1.gravity = params['gravity'] * rng(0.5, 1.5)

    elif env_name == 'mo-mountaincarcontinuous-v0':
        env1.power = params['power'] * rng(0.5, 1.5)

    elif env_name == 'mo-lunar-lander-v2':
        g = params['gravity'] * rng(0.5, 1.5)
        g = float(np.clip(g, -12.0, -1e-3)) 
        w = params['wind_power'] * rng(0.5, 1.5)
        w = float(np.clip(w, 0.0, 20.0))
        t = params['turbulence_power'] * rng(0.5, 1.5)
        t = float(np.clip(t, 0.0, 2.0))
        env1.gravity          = g
        env1.wind_power       = w
        env1.turbulence_power = t

    elif env_name == 'water-reservoir-v0':
        env1.nO         = int(round(params['nO'] * rng(0.5, 1.5)))
        env1.penalize   = bool(np.random.rand() < 0.5)
        env1.time_limit = int(round(params['time_limit'] * rng(0.5, 1.5)))

    elif env_name == 'resource-gathering-v0':
        size = np.random.choice([5, 7, 9], 1)[0]
        nr = 2 if size == 5 else np.random.choice([3, 4, 5, 6], 1)[0]
        # nr2 = 1 if size == 5 else np.random.choice([2, 4], 1)[0]
        ne = 2 if size == 5 else np.random.choice([3, 4], 1)[0]
        grid, init_pos = sample_resource_gathering_map(
            s=size,
            n_R=nr,
            n_E=ne,
        )
        env1.map = grid
        env1.initial_pos = init_pos

    elif env_name == 'four-room-v0':
        maze = sample_four_room_maze(
            n1=np.random.randint(3, 5),
            n2=np.random.randint(3, 5),
            n3=np.random.randint(3, 5),
            seed=np.random.randint(0, 10000),
        )
        env1.maze = maze
        env1.height, env1.width = maze.shape
        for c in range(env1.width):
            for r in range(env1.height):
                if maze[r, c] == "G":
                    env1.goal = (r, c)
                elif maze[r, c] == "_":
                    env1.initial.append((r, c))
                elif maze[r, c] == "X":
                    env1.occupied.add((r, c))
                elif maze[r, c] in {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}:
                    env1.shape_ids[(r, c)] = len(env1.shape_ids)
        env1.observation_space = Box(
            low=np.zeros(2 + len(env1.shape_ids)),
            high=len(env1.maze) * np.ones(2 + len(env1.shape_ids)),
            dtype=np.int32,
        )

    elif env_name == 'deep-sea-treasure-v0':
        sea_map, pareto_front = sample_treasures_from_map(
            noise_scale=0.5,
            seed=np.random.randint(0, 10000),
        )
        env1.sea_map = sea_map
        env1._pareto_front = pareto_front
    
    elif env_name == 'mo-highway-fast-v0':
        env1.config['lanes_count'] = np.random.randint(3,6)
        env1.config['vehicles_count'] = int(20 + np.random.normal(0, 3, 1))
        env1.config['controlled_vehicles'] = np.random.randint(1,3) 
        env1.config['ego_spacing'] = 1.5 + np.random.normal(0, 0.2, 1)
        
    return env1

    # tidak perlu return apa-apa; env sudah ter-update in-place


def get_parameter_env(env, env_name):
    if env_name == 'mo-halfcheetah-v4':
        return {
            '_forward_reward_weight': env._forward_reward_weight,
            '_ctrl_cost_weight': env._ctrl_cost_weight,
            '_reset_noise_scale': env._reset_noise_scale
        }
    elif env_name == 'mo-hopper-v4':
        return {
            "_forward_reward_weight": env._forward_reward_weight,
            "_ctrl_cost_weight": env._ctrl_cost_weight,
            "_healthy_reward": env._healthy_reward,
            "_terminate_when_unhealthy": env._terminate_when_unhealthy,
            "_healthy_state_range": env._healthy_state_range,
            "_healthy_z_range": env._healthy_z_range,
            "_healthy_angle_range": env._healthy_angle_range,
            "_reset_noise_scale": env._reset_noise_scale,
            }
    elif env_name == 'mo-mountaincar-v0':
        return {
            'force': env.force,
            'gravity': env.gravity
        }
    elif env_name == 'mo-mountaincarcontinuous-v0':
        return {
            'power': env.power
        }
    elif env_name == 'mo-lunar-lander-v2':
        return {
            'gravity': env.gravity, # gravity (current value: {gravity}) must be between -12 and 0
            'wind_power': env.wind_power, # 0.0 > wind_power value is recommended to be between 0.0 and 20.0
            'turbulence_power': env.turbulence_power # turbulence_power value is recommended to be between 0.0 and 2.0
        }
    elif env_name == 'water-reservoir-v0':
        return {
            'nO': env.nO,
            'penalize': env.penalize,
            'time_limit': env.time_limit
        }
    elif env_name == "resource-gathering-v0":
        return {
            'map': env.map,
            'initial_pos': env.initial_pos
        }
    elif env_name == 'four-room-v0':
        return {
            'maze': env.maze
        }
    elif env_name == "deep-sea-treasure-v0":
        return {
            'sea_map': env.sea_map,
            '_pareto_front': env._pareto_front 
        }
    elif env_name in ['mo-reacher-v4', 'mo-highway-fast-v0']:
        return {}

def sample_resource_gathering_map(s: int, n_R: int, n_E: int, rng=None):
    """
    Return (grid, initial_pos) for resource-gathering.
    grid: np.ndarray of shape (s, s) with " ", "H", "Rk", "Ek".
    initial_pos: (row, col) of H (bottom center).
    """
    if rng is None:
        rng = np.random.RandomState()

    # init grid with empty cells
    grid = np.full((s, s), " ", dtype=object)

    # place home H at bottom center
    h_row, h_col = s - 1, s // 2
    grid[h_row, h_col] = "H"
    initial_pos = (h_row, h_col)

    # all candidate positions except H
    all_cells = [(r, c) for r in range(s) for c in range(s)
                 if not (r == h_row and c == h_col)]
    rng.shuffle(all_cells)

    needed = n_R + n_E
    if needed > len(all_cells):
        raise ValueError("Too many R+E for grid size")

    # first n_R as resources
    for i in range(n_R):
        r, c = all_cells[i]
        grid[r, c] = f"R{(i%2) + 1}"

    offset = n_R
    for j in range(n_E):
        r, c = all_cells[offset + j]
        grid[r, c] = f"E{(j%2) + 1}"

    return grid, initial_pos


MAZE_TEMPLATE = np.array(
    [
        ["1", " ", " ", " ", " ", "2", "X", " ", " ", " ", " ", " ", "G"],
        [" ", " ", " ", " ", " ", " ", "X", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", "1", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", "X", " ", " ", " ", " ", " ", " "],
        ["2", " ", " ", " ", " ", "3", "X", " ", " ", " ", " ", " ", " "],
        ["X", "X", "3", " ", "X", "X", "X", "X", "X", " ", "1", "X", "X"],
        [" ", " ", " ", " ", " ", " ", "X", "2", " ", " ", " ", " ", "3"],
        [" ", " ", " ", " ", " ", " ", "X", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", "2", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", "X", " ", " ", " ", " ", " ", " "],
        ["_", " ", " ", " ", " ", " ", "X", "3", " ", " ", " ", " ", "1"],
    ],
    dtype=object,
)

def sample_four_room_maze(n1: int, n2: int, n3: int, seed: int = None):
    rng = np.random.RandomState(seed)
    maze = MAZE_TEMPLATE.copy()
    h, w = maze.shape

    for r in range(h):
        for c in range(w):
            if maze[r, c] != "X":
                maze[r, c] = " "

    def is_free(r, c):
        return maze[r, c] == " "

    corner_pairs = [((0, 0), (h-1, w-1)), ((0, w-1), (h-1, 0))]
    valid_pairs = []
    for (r1, c1), (r2, c2) in corner_pairs:
        if is_free(r1, c1) and is_free(r2, c2):
            valid_pairs.append(((r1, c1), (r2, c2)))
    if not valid_pairs:
        raise ValueError("No valid opposite corner pair for start/goal")

    (sr, sc), (gr, gc) = valid_pairs[rng.randint(len(valid_pairs))]

    maze[sr, sc] = "_"
    maze[gr, gc] = "G"
    start_pos = (sr, sc)
    goal_pos = (gr, gc)

    free_cells = [(r, c) for r in range(h) for c in range(w) if maze[r, c] == " "]
    rng.shuffle(free_cells)

    total_needed = n1 + n2 + n3
    if total_needed > len(free_cells):
        raise ValueError("Not enough free cells for digits")

    digit_pos = {"1": [], "2": [], "3": []}

    for _ in range(n1):
        r, c = free_cells.pop()
        maze[r, c] = "1"
        digit_pos["1"].append((r, c))

    for _ in range(n2):
        r, c = free_cells.pop()
        maze[r, c] = "2"
        digit_pos["2"].append((r, c))

    for _ in range(n3):
        r, c = free_cells.pop()
        maze[r, c] = "3"
        digit_pos["3"].append((r, c))

    return maze

import numpy as np

DEFAULT_MAP = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-10, 8.2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-10, -10, 11.5, 0, 0, 0, 0, 0, 0, 0, 0],
        [-10, -10, -10, 14.0, 15.1, 16.1, 0, 0, 0, 0, 0],
        [-10, -10, -10, -10, -10, -10, 0, 0, 0, 0, 0],
        [-10, -10, -10, -10, -10, -10, 0, 0, 0, 0, 0],
        [-10, -10, -10, -10, -10, -10, 19.6, 20.3, 0, 0, 0],
        [-10, -10, -10, -10, -10, -10, -10, -10, 0, 0, 0],
        [-10, -10, -10, -10, -10, -10, -10, -10, 22.4, 0, 0],
        [-10, -10, -10, -10, -10, -10, -10, -10, -10, 23.7, 0],
    ],
    dtype=float,
)

def sample_treasures_from_map(noise_scale: float = 0.1, seed: int | None = None):
    """
    Tambah noise ke nilai treasure:
      v_new = max(0, v + eps), eps ~ N(0, (noise_scale * v)^2)
    Return:
      grid: map dengan nilai treasure yang sudah di-noise
      treasures: list (row, col, value_noisy)
      convex_front: list np.array([value_noisy, time_obj])
    """
    rng = np.random.RandomState(seed)
    grid = DEFAULT_MAP.copy()
    h, w = grid.shape
    treasures = []

    for i in range(h):
        for j in range(w):
            v = grid[i, j]
            if v > 0:
                sigma = noise_scale * v
                eps = rng.normal(loc=0.0, scale=sigma)
                v_noisy = max(0.0, float(v + eps))
                grid[i, j] = np.round(v_noisy, 2)
                treasures.append((i, j, np.round(v_noisy, 2)))

    treasures.sort(key=lambda x: x[2])

    convex_front = []
    for k, (_, _, v) in enumerate(treasures):
        time_obj = -(1 + 2 * k)  # -1, -3, -5, ...
        convex_front.append(np.array([v, time_obj], dtype=float))

    return grid, convex_front

