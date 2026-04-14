# fine_tune_scalarized.py
# -*- coding: utf-8 -*-
# Environment-agnostic fine-tuning over multiple MO-Gymnasium tasks
# Requires: mo-gymnasium, gymnasium, ray[rllib], torch

import os, sys
import glob
import argparse
from typing import Dict, List, Optional
import tempfile

os.environ['PYTHONPATH'] = '/mnt/hum01-home01/p88346bn/scratch/project/merlion/script/metamorl/'
sys.path.append('/mnt/hum01-home01/p88346bn/scratch/project/merlion/script/metamorl/')
sys.path.append('/mnt/hum01-home01/p88346bn/scratch/project/merlion/script/')

os.environ['RAY_TMPDIR'] = '~/scratch/'
# os.environ['RAY_TMPDIR'] = '/mnt/hum01-home01/p88346bn/scratch/merlion_tmp/'
os.makedirs(os.environ['RAY_TMPDIR'], exist_ok=True)

import csv
import numpy as np
import ray
import time

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from ray.air import Checkpoint
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from algo_ray.rllib.algorithms.maml.maml_loc import MAMLConfigLoc
# Your wrapper env

from merlion._non_sc.envs.merlion_env_combine_notrandom import ScalarizedMetaEnv_notrandom, ScalarizationWeightProxy_notrandom
from itertools import combinations_with_replacement
from gymnasium.wrappers import TimeLimit

import json
import math

from ray.air import Checkpoint
from ray.rllib.algorithms.ppo import PPOConfig
try:
    from merlion.algo_ray.rllib.algorithms.maml import merlion_finetuning_general as ft
except:
    from algo_ray.rllib.algorithms.maml import merlion_finetuning_general as ft
    
def main():
    parser = argparse.ArgumentParser(description="Environment-agnostic fine-tuning over MO-Gymnasium tasks (from notebook last cell)")
    parser.add_argument("--id", type=int, default=1, help="1-based index into ENV_NAME; 1 -> 'mo-halfcheetah-v4'")
    parser.add_argument("--runs", type=int, default=1, help="Number of independent fine-tuning runs. now it is equal to seed number")
    args = parser.parse_args()

    # ========= ENV list (as in notebook) =========
    ENV_NAME: List[str] = [
        "mo-halfcheetah-v4",
        "deep-sea-treasure-v0",
        "resource-gathering-v0",
        "mo-mountaincar-v0",
        "mo-mountaincarcontinuous-v0",
        "mo-lunar-lander-v2",
        "water-reservoir-v0",
        'four-room-v0',
        'mo-highway-fast-v0',
        'mo-reacher-v4',
        'mo-hopper-v4',
    ]

    ## rollout_fragment_length
    H_CONFIG = {
        "mo-halfcheetah-v4": 1000,         
        "deep-sea-treasure-v0": 200,        
        "resource-gathering-v0": 200,        
        "mo-mountaincar-v0": 1000,           
        "mo-mountaincarcontinuous-v0": 1000, 
        "four-room-v0": 400,                
        "mo-highway-fast-v0": 200,         
        "mo-reacher-v4": 200,               
        "mo-hopper-v4": 1000,              
        "water-reservoir-v0": 200,          
        "mo-lunar-lander-v2": 1000,         
    }

    D_OBJ = {
        "mo-halfcheetah-v4": 2,           # [progress, ctrl_cost]
        "deep-sea-treasure-v0": 2,        # [treasure, time_penalty]
        "resource-gathering-v0": 3,       # [gold, diamond, death_penalty]
        "mo-mountaincar-v0": 3,           # [progress, energy]
        "mo-mountaincarcontinuous-v0": 2, # [progress, energy]
        "mo-lunar-lander-v2": 4,          # [outcome/crash, shaping, fuel_main, fuel_side]
        "water-reservoir-v0": 2,          # [supplied water, flood risk]
        "four-room-v0": 3,                # [goal, key, door] atau 3 collectible rewards
        "mo-highway-fast-v0": 3,          # [speed, right_lane, collision]
        "mo-reacher-v4": 4,               # [dist to 4 target spots]
        "mo-hopper-v4": 3                 # [progress, ctrl_cost]
    }

    # 1-based selection
    idx = args.id - 1
    if idx < 0 or idx >= len(ENV_NAME):
        raise ValueError(f"--id must be in [1..{len(ENV_NAME)}], got {args.id}")

    for env_id in ENV_NAME:
        register_env(
            f"Scalarized-{env_id}",
            lambda cfg=None, _env_id=env_id: TimeLimit(ScalarizedMetaEnv_notrandom({**(cfg or {}), "base_id": _env_id,
                                                                          'max_episode_steps': H_CONFIG[env_id]}), max_episode_steps=H_CONFIG[_env_id])
        )

    # ===== General knobs =====
    # === config ===
    ENV_CONF = dict(base_id=ENV_NAME[idx], reward_dim=D_OBJ[ENV_NAME[idx]], max_episode_steps=H_CONFIG[env_id])

    PPO_ITERS = math.ceil(100 / (H_CONFIG[ENV_NAME[idx]] / 1000))
    NUM_WORKERS = 4
    FRAG_LEN = H_CONFIG[ENV_NAME[idx]]
    EVAL_EPS = 5

    # Root for outputs and checkpoints
    # Fresh cluster for a clean registry (as in notebook)

    # IMPORTANT: restart Ray so workers pick it up
    MERLION_ROOT = "/mnt/hum01-home01/p88346bn/test/project/merlion/script/merlion/_non_sc/results/"
    MESSIAH_SRC  = "/mnt/hum01-home01/p88346bn/test/project/merlion/script/merlion/ds-project-messiah/src"

    ray.shutdown()
    ray.init(ignore_reinit_error=True,
    )

    # One canonical factory used by BOTH Ray + Gym.
    def make_scalarized(base_id: str, env_config: dict):
        cfg = dict(env_config or {})
        cfg.setdefault("base_id", base_id)
        cfg.setdefault("max_episode_steps", H_CONFIG[ENV_NAME[idx]])

        base = ScalarizedMetaEnv_notrandom(cfg)          # expects dict config
        base = TimeLimit(base, max_episode_steps=H_CONFIG[ENV_NAME[idx]])
        return ScalarizationWeightProxy_notrandom(base)            # exposes set_scalarization_weights

    for base_id in ENV_NAME:
        ray_name = f"Scalarized-{base_id}"

        # ---- RLlib registration ----
        def _creator(env_ctx, _base_id=base_id):
            return make_scalarized(_base_id, env_ctx)

        register_env(ray_name, _creator)

        # ---- Gymnasium registration (so gym.make(ray_name) works in your evaluator) ----
        try:
            gym.register(
                id=ray_name,
                entry_point=(lambda _base_id=base_id, **kwargs: make_scalarized(_base_id, kwargs)),
            )
        except Exception:
            # Already registered -> ignore
            pass

    for k in [args.runs]:
        start_time = time.time()
        ft.finetune_with_local_perturbations(
            pop_dir=f"/mnt/hum01-home01/p88346bn/scratch/project/merlion/script/merlion/_non_sc/results/{ENV_NAME[idx]}/metapolicies-{k}",
            env_id=f"Scalarized-{ENV_NAME[idx]}",
            out_dir=f"/mnt/hum01-home01/p88346bn/scratch/project/merlion/script/merlion/_non_sc/results/{ENV_NAME[idx]}/finetuning-{k}",
            total_steps= 100000, # default 5000
            record_every=H_CONFIG[ENV_NAME[idx]], # default 1
            eval_episodes=5, # see miracl_general.py and exist_general.py, they use 5
            m_perturb = 2 if D_OBJ[ENV_NAME[idx]] == 2 else 3, ## following the metamorl and miracl (21 and 32)
            num_objectives = D_OBJ[ENV_NAME[idx]],
            eps=0.05,
            seed=k,
            info_key='mo_reward_raw',
            save_file = False
        )
        end_time = time.time()
        print(f"fine-tuning run-{k}:",end_time - start_time,"seconds")

    ray.shutdown()
    print("\nALL FINE-TUNING DONE.")


if __name__ == "__main__":
    main()
