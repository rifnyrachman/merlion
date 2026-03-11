import os, sys, json
import argparse
from typing import Dict, List

# os.environ['PYTHONPATH'] = '/mnt/hum01-home01/p88346bn/test/project/merlion/script/merlion/'
# sys.path.append('/mnt/hum01-home01/p88346bn/test/project/merlion/script/merlion/')

import tempfile
os.environ['RAY_TMPDIR'] = '~/scratch/'
# os.environ['RAY_TMPDIR'] = '/mnt/hum01-home01/p88346bn/scratch/merlion_tmp/'
os.makedirs(os.environ['RAY_TMPDIR'], exist_ok=True)

from gymnasium.wrappers import TimeLimit
import gymnasium as gym
from pathlib import Path

# Import both modules
from merlion.algo_ray.rllib.algorithms.maml.merlion_main_multitasks import MERLION, MERLIONConfig
from merlion.algo_ray.rllib.algorithms.maml.callback_save_population import _find_archive , MERLIONWithSaveCallbacks
import math
#v2.3.1 #use algo from local repo
import ray
from ray import air, tune
from ray.tune.registry import register_env
from gymnasium.wrappers import TimeLimit
from ray.air import Checkpoint
from ray.rllib.algorithms.algorithm import Algorithm

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

import tempfile
import pickle
import json


from merlion.utils.plot_utils import PlotUtils
from merlion._non_sc.envs.merlion_env_combine import ScalarizedMetaEnv

# ENV list dari notebook
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
    "mo-halfcheetah-v4": 1000,          # locomotion panjang, standar MuJoCo [web:217]
    "deep-sea-treasure-v0": 200,         # grid kecil, treasure tercapai <~30 langkah [web:180]
    "resource-gathering-v0": 200,        # grid kecil, episode relatif pendek [web:191]
    "mo-mountaincar-v0": 1000,           # default max_episode_steps=200 [web:201][web:202]
    "mo-mountaincarcontinuous-v0": 1000, # default max_episode_steps=999 [web:210]
    "four-room-v0": 400,                # grid 4-room, perlu horizon agak panjang [web:186]
    "mo-highway-fast-v0": 200,           # duration ~40 step per episode [web:236]
    "mo-reacher-v4": 200,                # max_episode_steps=50 [web:222]
    "mo-hopper-v4": 1000,               # locomotion panjang, standar MuJoCo [web:217]
    "water-reservoir-v0": 200,          # horizon operasional moderat [web:233]
    "mo-lunar-lander-v2": 1000,         # max_episode_steps=1000 [web:106]
}

D_OBJ = {
    "mo-halfcheetah-v4": 2,           # [progress, ctrl_cost]
    "deep-sea-treasure-v0": 2,        # [treasure, time_penalty]
    "resource-gathering-v0": 3,       # [gold, diamond, death_penalty]
    "mo-mountaincar-v0": 3,           # [progress, energy]
    "mo-mountaincarcontinuous-v0": 2, # [progress, energy]
    "mo-lunar-lander-v2": 4,          # [outcome/crash, shaping, fuel_main, fuel_side]
    "water-reservoir-v0": 2,          # [supplied water, flood risk] (biasa 2 atau 4, cek config)
    "four-room-v0": 3,                # [goal, key, door] atau 3 collectible rewards
    "mo-highway-fast-v0": 3,          # [speed, right_lane, collision]
    "mo-reacher-v4": 4,               # [dist to 4 target spots]
    "mo-hopper-v4": 3                 # [progress, ctrl_cost]
}

TOTAL_TIMESTEPS = 3000000

def save_full_checkpoints_for_population(algo, out_dir: str):
    archive = _find_archive(algo)
    if archive is None:
        print("[SavePopulation] MERLION archive not found on algorithm; skipping.")
        return

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pol = algo.get_policy()
    P = int(getattr(archive, "population_size", len(getattr(archive, "archive", []))))

    for i in range(P):
        item = archive.get_archive_item(i)
        if item is None:
            continue

        theta_i = item["theta"]
        w_i = np.asarray(item["weight"], dtype=np.float32)   # <- correct key

        # save the full RLlib checkpoint for policy i
        pol.set_weights(theta_i)
        ckpt_dir = out / f"checkpoint_policy_{i}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = algo.save(str(ckpt_dir))
        print(f"[SavePopulation] Saved RLlib checkpoint for i={i} → {path}")

        # save the associated weight next to it
        with open(ckpt_dir / "weight.json", "w") as f:
            json.dump([float(x) for x in w_i.ravel().tolist()], f)

def register_scalarized_envs() -> None:
    # Registrasi "Scalarized-<base_id>" sesuai pola notebook
    for base_id in ENV_NAME:
        register_env(
            f"Scalarized-{base_id}",
            lambda cfg, _env_id=base_id: TimeLimit(ScalarizedMetaEnv({**(cfg or {}), "base_id": _env_id,
                                                            'max_episode_steps': H_CONFIG[_env_id]}), max_episode_steps=H_CONFIG[_env_id]),
        )

def build_merlion_config(env_name: str, num_workers: int, rollout_fragment_length: int, eval_interval: int, num_objectives: int = 2) -> MERLIONConfig:
    # Mencerminkan konfigurasi pada cell notebook (rollouts/training/evaluation)
    return (
        MERLIONConfig()
        .rollouts(
            num_rollout_workers=num_workers, rollout_fragment_length=rollout_fragment_length,
        )
        .framework("torch")
        .environment(f"Scalarized-{env_name}",
                    clip_actions=True,
                    )
        .training(
            inner_adaptation_steps=4,
            meta_batch_size=4,
            maml_optimizer_steps=10,
            gamma=0.99,
            lambda_=1.0,
            lr=0.001,
            vf_loss_coeff=0.5,
            inner_lr=0.03,
            use_meta_env=True,
            clip_param=0.3,
            kl_target=0.01,
            kl_coeff=0.001,
            model=dict(fcnet_hiddens=[64, 64]),
            train_batch_size=rollout_fragment_length,
            
            #MERLION specific parameters
            population_size = 10, #default
            num_objectives = num_objectives,
        )
        .evaluation(
            evaluation_num_workers=1,
            evaluation_interval=eval_interval,
            enable_async_evaluation=True,
        )
        .callbacks(MERLIONWithSaveCallbacks)
    )


def main():

    parser = argparse.ArgumentParser(description="CLI script derived from exist_combine.ipynb with Ray Tune Tuner")
    parser.add_argument("--id", type=int, default=1, help="1-based index into ENV_NAME; 1 -> mo-halfcheetah-v4")
    parser.add_argument("--runs", type=int, default=1, help="now it is equal to seed number")
    # parser.add_argument("--train-iters", type=int, default=5, help="Maximum training_iteration per trial (stop condition)")
    # parser.add_argument("--num-samples", type=int, default=1, help="Number of trials (configurations) for Tune")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of rollout workers")
    # parser.add_argument("--rollout-fragment-length", type=int, default=32, help="Rollout fragment length")
    # parser.add_argument("--eval-interval", type=int, default=100, help="Evaluation interval (iterations)")
    
    args = parser.parse_args()

    # Validasi dan pemilihan env berbasis 1
    idx = args.id - 1
    if idx < 0 or idx >= len(ENV_NAME):
        raise ValueError(f"--id harus di [1..{len(ENV_NAME)}], dapat {args.id}")
    base_env = ENV_NAME[idx]
    env_name = f"Scalarized-{base_env}"
    print(f'running in {env_name}')

    # Ray init dan env registration 
    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    register_scalarized_envs()

    inner_adaptation_steps = 4
    meta_batch_size = 4

    ### P (J (S+1)H + H)
    ### J = meta batch size
    num_iterations = math.ceil(TOTAL_TIMESTEPS / (10 * meta_batch_size * (inner_adaptation_steps+1) * H_CONFIG[base_env]))

    # build RLlib AlgorithmConfig (MAML)
    algo_config = build_merlion_config(
        env_name=base_env,
        num_workers=args.num_workers,
        rollout_fragment_length=H_CONFIG[base_env],
        eval_interval= int(num_iterations // 2),
        num_objectives=D_OBJ[base_env]
    )

    local_dir = f"/mnt/hum01-home01/p88346bn/scratch/project/merlion/script/merlion/_non_sc/results/{base_env}"

    print(f'running merlion for {num_iterations} iteration')
    results = {}
    runs = args.runs

    # CHANGED: ensure base directory exists
    os.makedirs(local_dir, exist_ok=True)

    for k in [runs]:
        print(f"RUNNING NUMBER-{k}")

        # CHANGED: pre-create the experiment directory (avoids .tmp_checkpoint path errors)

        # exp_dir = os.path.join(local_dir, f"merlion_{k}")
        # os.makedirs(exp_dir, exist_ok=True)

        tuner = tune.Tuner(
            MERLION,
            param_space=algo_config.to_dict(),
            run_config=air.RunConfig(
                name=f"merlion_{base_env}_{k}",
                local_dir=local_dir,  # keep as-is
                stop={"training_iteration": num_iterations},
                failure_config=air.FailureConfig(fail_fast="raise"),  # keep as-is
                # CHANGED: explicit log files (avoids odd __stdout_file__ fields)
                log_to_file=True,
                # ("stdout.log", "stderr.log"),
                checkpoint_config=air.CheckpointConfig(
                    checkpoint_at_end=True,
                    num_to_keep=1
                ),
            )
        )

        results[k] = tuner.fit()

        # CHANGED: if trial didn’t complete, skip checkpoint restore gracefully
        # if results[k] is None or results[k].num_results == 0:
        #     print(f"[WARN] No completed trials for run {k}. You can resume with:")
        #     print(f"       tuner = tune.Tuner.restore('{exp_dir}', trainable=MERLION)")
        #     continue

        # save the best checkpoint for each run
        best = results[k].get_best_result()  # (optionally pass metric=..., mode=...)
        if best.checkpoint is None:
            print(f"[WARN] Best result for run {k} has no checkpoint; skipping export.")
            continue

        algo = Algorithm.from_checkpoint(best.checkpoint)

        out_dir = os.path.join(local_dir, f"metapolicies-{k}")
        os.makedirs(out_dir, exist_ok=True)
        save_full_checkpoints_for_population(algo, out_dir)
        algo.stop()
        
if __name__ == "__main__":
    main()
