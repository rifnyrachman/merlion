import sys, os
import logging
import numpy as np
from typing import Optional, Type, List, Tuple, Dict, Any, Union
import random
import copy

from .merlion_utils import (
    project_simplex, generate_uniform_weights, generate_das_dennis_weights,
    pairwise_log_distance_entropy, marginal_hv_contribution, farthest_index
)

logger = logging.getLogger(__name__)

# ==========================
# Archive (Algorithms 1&2)
# ==========================

def _coerce_objective_vector(x, M: int) -> np.ndarray:
    """
    Accepts:
      - None -> returns None
      - scalar -> pads to length M (scalar in index 0)
      - 1D -> trims/pads to length M
      - 2D (T x M') -> reduces over T (sum) and trims/pads to M
      - dict with common keys -> picks first found and recurses
    Returns float32 array (M,) or None.
    """
    if x is None:
        return None
    if isinstance(x, dict):
        for k in ("mo_reward_vec", "r_vec_used", "r_vec_oriented", "r_vec_raw", "objectives"):
            if k in x:
                return _coerce_objective_vector(x[k], M)
        return None

    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 0:
        out = np.zeros(M, dtype=np.float32); out[0] = float(arr)
        return out
    if arr.ndim == 1:
        if arr.size >= M:
            return arr[:M].astype(np.float32, copy=False)
        out = np.zeros(M, dtype=np.float32)
        out[:arr.size] = arr
        return out
    if arr.ndim == 2:
        # Per-step or per-episode rows → sum over rows (use mean if you prefer)
        arr = arr[:, :M]
        return arr.sum(axis=0).astype(np.float32, copy=False)
    # Anything else → give up.
    return None

def _is_allclose_zero(v: np.ndarray) -> bool:
    return v is None or not np.isfinite(v).all() or np.allclose(v, 0.0, atol=1e-7)

class MERLIONArchive:
    """Stores population of (θᵢ, wᵢ) pairs and handles evolution (Algorithm 2)."""

    def __init__(self, population_size: int, num_objectives: int,
                 initial_theta: Union[dict, List[dict]]):
        self.population_size = population_size
        self.num_objectives = num_objectives
        self.archive: List[Dict[str, Any]] = []
        self.population_weights = generate_uniform_weights(population_size, num_objectives) # options: generate_uniform_weights, generate_das_dennis_weights
        self.iteration_count = 0

        if isinstance(initial_theta, list):
            assert len(initial_theta) == population_size, \
                "initial_theta list length must equal population_size"
            for i in range(population_size):
                theta_i = copy.deepcopy(initial_theta[i])  # safety: avoid aliasing
                self.archive.append({
                    'theta': theta_i,
                    'weight': self.population_weights[i].copy(),
                    'fitness': 0.0,
                    'objectives': np.zeros(self.num_objectives, dtype=np.float32),
                    'adapted_theta': None,
                    'scalarized_value': 0.0,
                })
        else:
            # Backward-compatible: clone the single θ across the population
            for i in range(population_size):
                theta_i = copy.deepcopy(initial_theta)
                self.archive.append({
                    'theta': theta_i,
                    'weight': self.population_weights[i].copy(),
                    'fitness': 0.0,
                    'objectives': np.zeros(self.num_objectives, dtype=np.float32),
                    'adapted_theta': None,
                    'scalarized_value': 0.0,
                })

        logger.info(f"Initialized archive with {len(self.archive)} meta-policies")
    # --- Getters ---
    def get_population(self) -> List[Tuple[dict, np.ndarray]]:
        return [(it['theta'], it['weight']) for it in self.archive]

    def get_archive_item(self, index: int):
        if 0 <= index < len(self.archive):
            return self.archive[index]
        return None

    def get_weights_array(self) -> np.ndarray:
        return np.array([it['weight'] for it in self.archive], dtype=np.float32)

    # --- Updaters ---
    def update_archive(self, theta_i: dict, weight_i: np.ndarray, index: int, adapted_theta: dict = None):
        if 0 <= index < len(self.archive):
            self.archive[index]['theta'] = copy.deepcopy(theta_i)
            self.archive[index]['weight'] = project_simplex(np.asarray(weight_i))
            if adapted_theta is not None:
                self.archive[index]['adapted_theta'] = copy.deepcopy(adapted_theta)

#     def update_archive_with_objectives(
#         self, theta_i: dict, weight_i: np.ndarray, index: int,
#         objective_values: np.ndarray = None, adapted_theta: dict = None
#     ):
#         self.update_archive(theta_i, weight_i, index, adapted_theta)
#         # if 0 <= index < len(self.archive) and objective_values is not None:
#         #     obj = np.asarray(objective_values, dtype=np.float32)[:self.num_objectives]
#         #     self.archive[index]['objectives'] = obj
#         #     self.archive[index]['scalarized_value'] = float(np.dot(obj, self.archive[index]['weight']))

#         if 0 <= index < len(self.archive) and objective_values is not None:
#             obj = np.asarray(objective_values, dtype=np.float32)[:self.num_objectives]
#             self.archive[index]['objectives'] = obj
#             sc = float(np.dot(obj, self.archive[index]['weight']))
#             self.archive[index]['scalarized_value'] = sc
#             # NEW: keep a fitness number usable elsewhere (rep selection, logs, etc.)
#             self.archive[index]['fitness'] = sc

    def update_archive_with_objectives(
        self, theta_i: dict, weight_i: np.ndarray, index: int,
        objective_values: Any = None, adapted_theta: dict = None,
        allow_zero_overwrite: bool = False, reduce="sum"
    ):
        """
        objective_values may be:
          - vector shape (M,)
          - matrix shape (T, M)  -> reduced along T (sum/mean)
          - scalar                -> placed in dim 0 with zero-pad
          - dict with 'mo_reward_vec' / 'r_vec_used' / etc.
        We DO NOT overwrite with all-zeros unless allow_zero_overwrite=True.
        """
        self.update_archive(theta_i, weight_i, index, adapted_theta)

        if not (0 <= index < len(self.archive)):
            return

        # coerce and reduce
        M = self.num_objectives
        obj = _coerce_objective_vector(objective_values, M)

        # optional mean instead of sum
        if isinstance(objective_values, (list, tuple, np.ndarray)):
            arr = np.asarray(objective_values, dtype=np.float32)
            if arr.ndim == 2 and reduce == "mean":
                obj = arr[:, :M].mean(axis=0).astype(np.float32, copy=False)

        # refuse to clobber with zeros/NaN unless explicitly allowed
        if _is_allclose_zero(obj) and not allow_zero_overwrite:
            # keep previous objectives, but still update weight/theta
            prev = self.archive[index].get('objectives', np.zeros(M, np.float32))
            self.archive[index]['scalarized_value'] = float(np.dot(prev, self.archive[index]['weight']))
            # lightweight debug breadcrumb
            logger.warning(f"[Archive] Skipped zero/invalid objectives for idx={index}; "
                           f"caller passed type={type(objective_values)} shape="
                           f"{getattr(np.asarray(objective_values, dtype=np.float32), 'shape', None)}")
            return

        # commit
        self.archive[index]['objectives'] = np.asarray(obj, np.float32)
        sc = float(np.dot(self.archive[index]['objectives'], self.archive[index]['weight']))
        self.archive[index]['scalarized_value'] = sc
        self.archive[index]['fitness'] = sc


    # --- Monitoring ---
    # def print_population_objectives(self): # hard coded for 3 objectives
    #     print(f"\n=== POPULATION OBJECTIVE BREAKDOWN (Iter {self.iteration_count}) ===")
    #     for i, it in enumerate(self.archive):
    #         w = it['weight']; o = it['objectives']; s = it['scalarized_value']
    #         print(f"{i:02d} w=[{w[0]:.3f},{w[1]:.3f},{w[2]:.3f}] "
    #               f"obj=[{o[0]:7.2f},{o[1]:7.2f},{o[2]:7.2f}] sc={s:9.3f}")
    #     print("="*64)
    
    def print_population_objectives(self):
        M = int(self.num_objectives)
        print(f"\n=== POPULATION OBJECTIVE BREAKDOWN (Iter {self.iteration_count}) ===")
        for i, it in enumerate(self.archive):
            w = np.asarray(it['weight'], dtype=np.float32).ravel()[:M]
            o = np.asarray(it['objectives'], dtype=np.float32).ravel()[:M]
            s = float(it.get('scalarized_value', 0.0))
            w_str = ",".join(f"{x:.3f}" for x in w)
            o_str = ",".join(f"{x:7.2f}" for x in o)
            flag = "ZERO!" if _is_allclose_zero(o) else "OK"
            print(f"{i:02d} w=[{w_str}] obj=[{o_str}] sc={s:9.3f}  <- {flag}")
        print("=" * 64)
        
        #TODO: add export function to store rewards and meta-policies per individual solutions

    # --- Evolution (Algorithm 2) ---
#     def evolve_weights(self, eta_cross: float = 0.6, delta: float = 0.1, alpha: float = 0.5):
#         """
#         Algorithm 2: Fitness Q(w_i) = α·ΔHV(w_i) + (1−α)·ΔE(w_i).
#         - Parent1: rank-based selection on Q
#         - Parent2: farthest weight vector from Parent1
#         - Crossover (prob eta_cross): arithmetic interpolation
#         - Else mutation: add uniform noise in [-delta, delta]^b
#         - Project to simplex
#         - θ inheritance: copy θ from fitter parent
#         - Return a new population of size P
#         """
#         try:
#             P = self.population_size
#             W = self.get_weights_array()
#             # Use current OBJECTIVES for HV; maximize all 3 (P, -E, -I)
#             O = np.array([it['objectives'] for it in self.archive], dtype=np.float32)
#             if len(O) != P:
#                 O = np.zeros((P, self.num_objectives), dtype=np.float32)
#             # Reference point slightly below min observed
#             ref = (O.min(axis=0, initial=0.0) - 1.0).astype(np.float32)
#             dHV = marginal_hv_contribution(O, ref)  # [P]
#             dE = pairwise_log_distance_entropy(W)   # [P]
#             Q = alpha * dHV + (1.0 - alpha) * dE

#             # Rank-based selection probabilities
#             order = np.argsort(-Q)  # best first
#             ranks = np.empty_like(order); ranks[order] = np.arange(P)
#             sel_prob = (P - ranks) / np.sum(P - ranks)

#             def pick_parent1() -> int:
#                 return int(np.random.choice(P, p=sel_prob))

#             def pick_parent2(p1: int) -> int:
#                 return farthest_index(W, p1)

#             def crossover(wi: np.ndarray, wj: np.ndarray) -> np.ndarray:
#                 lam = float(np.random.uniform(0.1, 0.9))
#                 return project_simplex(lam * wi + (1.0 - lam) * wj)

#             def mutate(w: np.ndarray) -> np.ndarray:
#                 noise = np.random.uniform(-delta, delta, size=w.shape).astype(np.float32)
#                 return project_simplex(w + noise)

#             new_pop: List[Dict[str, Any]] = []
#             for _ in range(P):
#                 i = pick_parent1()
#                 j = pick_parent2(i)
#                 wi, wj = W[i], W[j]

#                 if np.random.rand() < eta_cross:
#                     w_new = crossover(wi, wj)
#                 else:
#                     w_new = mutate(wi)

#                 # θ inheritance: take fitter parent's θ
#                 par = i if Q[i] >= Q[j] else j
#                 theta_new = copy.deepcopy(self.archive[par]['theta'])

#                 new_pop.append({
#                     'theta': theta_new,
#                     'weight': w_new,
#                     'fitness': 0.0,
#                     'objectives': np.zeros(self.num_objectives, dtype=np.float32),
#                     'adapted_theta': None,
#                     'scalarized_value': 0.0,
#                 })

#             self.archive = new_pop
#             self.iteration_count += 1
#             logger.info("Evolutionary procedure completed (Algorithm 2)")
#         except Exception as e:
#             logger.error(f"Evolutionary procedure failed: {e}")

    def evolve_weights(self, eta_cross: float = 0.6, delta: float = 0.1,
                       alpha: float = 0.5, elite_k: int = 1):
        P = self.population_size
        W = self.get_weights_array()
        O = np.array([it['objectives'] for it in self.archive], dtype=np.float32)
        if len(O) != P:
            O = np.zeros((P, self.num_objectives), dtype=np.float32)
        ref = (O.min(axis=0, initial=0.0) - 1.0).astype(np.float32)
        dHV = marginal_hv_contribution(O, ref)
        dE  = pairwise_log_distance_entropy(W)
        Q   = alpha * dHV + (1.0 - alpha) * dE

        # Update stored fitness so other parts can see it (optional but helpful)
        for i in range(P):
            self.archive[i]['fitness'] = float(Q[i])

        # ----- ELITISM -----
        elite_k = max(0, min(elite_k, P))
        elite_idx = np.argsort(-Q)[:elite_k]
        new_pop: List[Dict[str, Any]] = []
        for idx in elite_idx:
            # copy best individuals unchanged
            it = self.archive[idx]
            new_pop.append({
                'theta': copy.deepcopy(it['theta']),
                'weight': it['weight'].copy(),
                'fitness': float(Q[idx]),
                'objectives': it['objectives'].copy(),
                'adapted_theta': copy.deepcopy(it['adapted_theta']),
                'scalarized_value': float(np.dot(it['objectives'], it['weight'])),
            })
        # -------------------

        # Rank-based selection over the rest
        order = np.argsort(-Q)
        ranks = np.empty_like(order); ranks[order] = np.arange(P)
        sel_prob = (P - ranks) / np.sum(P - ranks)

        def pick_parent1() -> int:
            return int(np.random.choice(P, p=sel_prob))

        def pick_parent2(p1: int) -> int:
            return farthest_index(W, p1)

        def crossover(wi: np.ndarray, wj: np.ndarray) -> np.ndarray:
            lam = float(np.random.uniform(0.1, 0.9))
            return project_simplex(lam * wi + (1.0 - lam) * wj)

        def mutate(w: np.ndarray) -> np.ndarray:
            noise = np.random.uniform(-delta, delta, size=w.shape).astype(np.float32)
            return project_simplex(w + noise)

        while len(new_pop) < P:
            i = pick_parent1()
            j = pick_parent2(i)
            wi, wj = W[i], W[j]
            w_new = crossover(wi, wj) if np.random.rand() < eta_cross else mutate(wi)
            par = i if Q[i] >= Q[j] else j  # θ inheritance from fitter parent
            theta_new = copy.deepcopy(self.archive[par]['theta'])
            new_pop.append({
                'theta': theta_new,
                'weight': w_new,
                'fitness': 0.0,
                'objectives': np.zeros(self.num_objectives, dtype=np.float32),
                'adapted_theta': None,
                'scalarized_value': 0.0,
            })

        self.archive = new_pop
        self.iteration_count += 1
        logger.info("Evolutionary procedure completed (Algorithm 2 + elitism)")
        
        # ---- serialization helpers ----
        def to_dict(self) -> dict:
            pop = []
            for it in self.archive:
                pop.append({
                    "theta": it["theta"],
                    "weight": np.asarray(it["weight"], np.float32).tolist(),
                    "fitness": float(it.get("fitness", 0.0)),
                    "objectives": np.asarray(
                        it.get("objectives", np.zeros(self.num_objectives, np.float32)),
                        np.float32,
                    ).tolist(),
                    "adapted_theta": it.get("adapted_theta"),
                    "scalarized_value": float(it.get("scalarized_value", 0.0)),
                })
            return {
                "population_size": int(self.population_size),
                "num_objectives": int(self.num_objectives),
                "iteration_count": int(getattr(self, "iteration_count", 0)),
                "population": pop,
            }

        @classmethod
        def from_dict(cls, data: dict) -> "MERLIONArchive":
            P = int(data["population_size"])
            M = int(data["num_objectives"])
            pop = data["population"]

            # build shell archive (constructor signature: adjust if needed)
            init_thetas = [it["theta"] for it in pop]
            arch = cls(P, M, init_thetas)
            arch.iteration_count = int(data.get("iteration_count", 0))

            rebuilt = []
            for it in pop:
                rebuilt.append({
                    "theta": copy.deepcopy(it["theta"]),
                    "weight": np.asarray(it["weight"], np.float32),
                    "fitness": float(it.get("fitness", 0.0)),
                    "objectives": np.asarray(it.get("objectives", np.zeros(M, np.float32)), np.float32),
                    "adapted_theta": copy.deepcopy(it.get("adapted_theta")),
                    "scalarized_value": float(it.get("scalarized_value", 0.0)),
                })
            arch.archive = rebuilt
            arch.population_weights = np.stack([it["weight"] for it in rebuilt], axis=0)
            return arch

        def save(self, path: str):
            import pickle
            with open(path, "wb") as f:
                pickle.dump(self.to_dict(), f, protocol=pickle.HIGHEST_PROTOCOL)

        @classmethod
        def load(cls, path: str) -> "MERLIONArchive":
            import pickle
            with open(path, "rb") as f:
                data = pickle.load(f)
            return cls.from_dict(data)
