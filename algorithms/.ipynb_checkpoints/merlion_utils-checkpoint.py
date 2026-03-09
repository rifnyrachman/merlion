import sys, os
import logging
import numpy as np
from typing import Optional, Type, List, Tuple, Dict, Any
import random
import copy
import torch
import torch.nn.init as init
import math
from itertools import combinations

# ===========================
# Helpers (Simplex & Metrics)
# ===========================

def project_simplex(w: np.ndarray) -> np.ndarray:
    """Project a vector to the probability simplex {w >= 0, sum w = 1}."""
    w = np.asarray(w, dtype=np.float32).copy()
    w[w < 0.0] = 0.0
    s = w.sum()
    if s <= 0.0:
        return np.full_like(w, 1.0 / len(w))
    return w / s


def pairwise_log_distance_entropy(W: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Entropy-like diversity contribution for each weight vector:
    ΔE_i = average_j log(||w_i - w_j|| + eps).
    Returns per-individual scores (higher is more diverse).
    """
    P = W.shape[0]
    out = np.zeros(P, dtype=np.float32)
    for i in range(P):
        d = np.linalg.norm(W[i] - W, axis=1)
        d[i] = 0.0
        out[i] = float(np.mean(np.log(d + eps)))
    return out


def hypervolume(points: np.ndarray, ref: np.ndarray) -> float:
    """
    Very lightweight HV approximation for small b (b=3).
    Assumes we are maximizing each objective and ref is a dominated reference point.
    For speed/robustness here we use Monte Carlo grid approximation.
    """
    pts = np.asarray(points, dtype=np.float32)
    b = pts.shape[1]
    # Normalize to [0,1] w.r.t. ref and max over points (avoid infinite ref)
    maxs = np.maximum(pts.max(axis=0), ref + 1e-6)
    mins = ref
    mins = np.minimum(mins, pts.min(axis=0))

    # Sample grid
    K = 2000  # balance speed/quality
    us = np.random.rand(K, b) * (maxs - mins) + mins
    # Check if dominated by at least one point
    dominated = np.zeros(K, dtype=bool)
    for p in pts:
        dominated |= np.all(us <= p, axis=1)
    hv_est = dominated.mean() * np.prod(maxs - mins)
    return float(hv_est)


def marginal_hv_contribution(points: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    ΔHV_i = HV(P) - HV(P \ {i}). Uses the same HV approximation for consistency.
    """
    P = points.shape[0]
    hv_all = hypervolume(points, ref)
    contrib = np.zeros(P, dtype=np.float32)
    for i in range(P):
        mask = np.ones(P, dtype=bool); mask[i] = False
        hv_wo = hypervolume(points[mask], ref)
        contrib[i] = float(hv_all - hv_wo)
    return contrib


def farthest_index(W: np.ndarray, i: int) -> int:
    """Index of the farthest weight vector from w_i under Euclidean distance."""
    d = np.linalg.norm(W - W[i], axis=1)
    d[i] = -1.0
    return int(np.argmax(d))


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if a Pareto-dominates b (assume maximize all)."""
    return np.all(a >= b) and np.any(a > b)

# ==========================
# Weight initialisation
# ==========================

def generate_uniform_weights(population_size: int, num_objectives: int) -> np.ndarray:
    """Generate weight vectors on the simplex via Dirichlet(1,...,1)."""
    alpha = np.ones(num_objectives, dtype=np.float32)
    W = np.random.dirichlet(alpha, size=population_size).astype(np.float32)
    return W

def generate_das_dennis_weights(
    population_size: int,
    num_objectives: int,
    H: int | None = None,
    subsample: bool = True,
) -> np.ndarray:
    """
    Generate (approximately) `population_size` evenly-spaced weight vectors
    on the simplex using the Das–Dennis construction.

    Args:
        population_size: target number of weight vectors you want.
        num_objectives:  number of objectives M.
        H:               optional division parameter. If None, we choose the
                         smallest H such that comb(H + M - 1, M - 1) >= population_size.
        subsample:       if True and the grid is larger than population_size,
                         randomly subsample down to population_size.

    Returns:
        W: array of shape (K, num_objectives), with K >= population_size if
           subsample=False, or K == population_size if subsample=True.
    """
    M = num_objectives

    # 1) Choose H if not provided
    if H is None:
        H = 1
        while math.comb(H + M - 1, M - 1) < population_size:
            H += 1

    # Number of grid points for this H
    grid_size = math.comb(H + M - 1, M - 1)
    # print(f"[Das-Dennis] M={M}, H={H}, grid_size={grid_size}")

    # 2) Generate all integer tuples (a1,...,a_M) >= 0 with sum ai = H
    #    using the stars-and-bars combinatorics via combinations.
    all_weights = []
    for cuts in combinations(range(H + M - 1), M - 1):
        # Add sentinels to get segment lengths
        cuts = (-1,) + cuts + (H + M - 1,)
        a = [cuts[i + 1] - cuts[i] - 1 for i in range(M)]  # each a_i >= 0, sum a_i = H
        w = np.array(a, dtype=np.float32) / float(H)       # normalize to sum 1
        all_weights.append(w)

    W = np.stack(all_weights, axis=0).astype(np.float32)

    # 3) Adjust size if requested
    if subsample and W.shape[0] > population_size:
        idx = np.random.choice(W.shape[0], size=population_size, replace=False)
        W = W[idx]

    return W


def scalarize_reward(multi_obj_reward: np.ndarray, weight_vector: np.ndarray) -> float:
    return float(np.dot(multi_obj_reward, weight_vector))

# ============================
# Meta-policies Initialization
# ============================

def _add_gaussian_noise(weights, std=1.0, per_key_std=None, rng=None): #std=0.02
    """
    Add Gaussian noise to a policy weights dict (Torch/NumPy values supported).
    Returns a new dict with the same key structure and NumPy arrays for numeric params.
    """
    if rng is None:
        rng = np.random.RandomState()
    noisy = {}
    for k, v in weights.items():
        if isinstance(v, torch.Tensor):
            arr = v.detach().cpu().numpy()
        elif isinstance(v, np.ndarray):
            arr = v
        else:
            noisy[k] = v
            continue
        sigma = (per_key_std.get(k, std) if per_key_std else std)
        noisy[k] = arr + rng.randn(*arr.shape).astype(arr.dtype) * sigma
        #print(f"noisy-{k}:",noisy[k])
    return noisy

def generate_diverse_meta_policies(
    policy_cls, obs_space, act_space, pol_cfg, population_size,
    diversity_method="mixed", noise_std=0.02, per_key_std=None # default noise_std=0.02
):
    initial_thetas = []
    for i in range(population_size):
        # Per-individual RNG/seed
        if diversity_method == "seed":
            seed = i * 1000 + np.random.randint(0, 1000)
            import random
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState(i * 9973 + 123)

        tmp_pol = policy_cls(obs_space, act_space, pol_cfg)
        try:
            weights = tmp_pol.get_weights()

            if diversity_method in ["scale", "mixed"]:
                # --- compute scale_factor here ---
                if diversity_method == "scale":
                    # range ~[0.1, 2.0]
                    denom = max(1, population_size - 1)
                    scale_factor = 0.1 + (i / denom) * 1.9
                else:  # "mixed"
                    # thirds: small / normal / large
                    one_third = population_size // 3
                    two_thirds = 2 * one_third
                    if i < one_third:
                        # small: 0.1, 0.2, 0.3 cycling
                        scale_factor = 0.1 + (i % 3) * 0.1
                    elif i < two_thirds:
                        # normal: 0.5, 0.7, 0.9 cycling
                        scale_factor = 0.5 + ((i - one_third) % 3) * 0.2
                    else:
                        # large: 1.0, 1.5, 2.0 cycling
                        scale_factor = 1.0 + ((i - two_thirds) % 3) * 0.5

                # apply scaling, keeping dtypes
                scaled = {}
                for k, v in weights.items():
                    if isinstance(v, torch.Tensor):
                        scaled[k] = v * scale_factor
                    elif isinstance(v, np.ndarray):
                        scaled[k] = v * scale_factor
                    else:
                        scaled[k] = v
                tmp_pol.set_weights(scaled)
                weights = tmp_pol.get_weights()

            # add a little gaussian noise to break symmetry
            weights = _add_gaussian_noise(
                weights, std=noise_std, per_key_std=per_key_std, rng=rng
            )
            initial_thetas.append(weights)
        finally:
            del tmp_pol

    return initial_thetas

def generate_orthogonal_meta_policies(policy_cls, obs_space, act_space, pol_cfg, population_size):
    """
    Generate meta-policies with orthogonal weight initialization for maximum diversity.
    """
    initial_thetas = []
    
    for i in range(population_size):
        tmp_pol = policy_cls(obs_space, act_space, pol_cfg)
        
        try:
            # Apply orthogonal initialization to all linear layers
            for module in tmp_pol.model.modules():
                if isinstance(module, torch.nn.Linear):
                    # Use different gain values for diversity
                    gain = 0.5 + (i / population_size) * 1.5  # Range: 0.5 to 2.0
                    init.orthogonal_(module.weight, gain=gain)
                    if module.bias is not None:
                        init.constant_(module.bias, 0.1 * (i / population_size))
            
            initial_thetas.append(tmp_pol.get_weights())
            
        finally:
            del tmp_pol
    
    return initial_thetas

def generate_strategy_based_meta_policies(policy_cls, obs_space, act_space, pol_cfg, 
                                        population_size):
    """
    Generate meta-policies with different behavioral strategies.
    """
    initial_thetas = []
    strategies = ["conservative", "aggressive", "exploratory", "exploitative"]
    
    for i in range(population_size):
        strategy = strategies[i % len(strategies)]
        tmp_pol = policy_cls(obs_space, act_space, pol_cfg)
        
        try:
            for module in tmp_pol.model.modules():
                if isinstance(module, torch.nn.Linear):
                    if strategy == "conservative":
                        # Small weights, low variance
                        init.normal_(module.weight, mean=0.0, std=0.1)
                    elif strategy == "aggressive":
                        # Large weights, high variance
                        init.normal_(module.weight, mean=0.0, std=0.5)
                    elif strategy == "exploratory":
                        # Uniform distribution for exploration
                        init.uniform_(module.weight, -0.5, 0.5)
                    elif strategy == "exploitative":
                        # Xavier initialization for stable learning
                        init.xavier_uniform_(module.weight)
                    
                    if module.bias is not None:
                        bias_val = {"conservative": -0.1, "aggressive": 0.1,
                                  "exploratory": 0.0, "exploitative": 0.0}[strategy]
                        init.constant_(module.bias, bias_val)
            
            initial_thetas.append(tmp_pol.get_weights())
            
        finally:
            del tmp_pol
    
    return initial_thetas
