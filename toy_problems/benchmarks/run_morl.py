# import os

# custom_temp_dir = 'atmp/morl_tmp/'

# # 2. Buat folder dulu
# try:
#     os.makedirs(custom_temp_dir, exist_ok=True)
#     os.makedirs(os.path.join(custom_temp_dir, "wandb_cache"), exist_ok=True)
# except OSError as e:
#     print(f"Gagal membuat folder: {e}")

# # 3. Set environment variables SEBELUM import library lain
# os.environ["TMPDIR"] = custom_temp_dir
# os.environ["TEMP"] = custom_temp_dir
# os.environ["TMP"] = custom_temp_dir
# os.environ["WANDB_DIR"] = custom_temp_dir
# os.environ["WANDB_CACHE_DIR"] = os.path.join(custom_temp_dir, "wandb_cache")

import argparse
import numpy as np
import mo_gymnasium as mo_gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
# from gymnasium.wrappers import FlattenObservation, RecordVideo
# from mo_gymnasium.wrappers import RecordEpisodeStatistics

from benchmark_algorithms_call import get_algorithms, ENV_NAME

if __name__ == '__main__':
    # env = mo_gym.make(env_name)
    # ref_point = np.array([0.0]*NUM_OBJ[env_name])

    parser = argparse.ArgumentParser(description='run_rl')
    parser.add_argument('--env_name', type=str, default='deep-sea-treasure-v0', help='Environment name.')
    parser.add_argument('--env_id', type=int, default=0, help='Environment ID.')
    parser.add_argument('--method', type=str, default='morld', help='Method to use.')
    parser.add_argument('--ts', type=int, default=3000000, help='Timesteps.')
    # parser.add_argument('--ts_it', type=int, default=1000, help='Timesteps per iteration.')
    args = parser.parse_args()
    
    if args.env_id > 0: ## if it is index not zero 
        args.env_name = ENV_NAME[args.env_id-1]

    runs = range(0,10)
    for seed in runs:
        print(f'training {args.method} on environment {args.env_name} on seed {seed} for {args.ts} steps')
        agent, eval_env, ref_point = get_algorithms(args.env_name, args.method, seed)

        # if not os.path.exists(f'{args.env_name}'):
        #     os.makedirs(f'{args.env_name}')
        # print(f"Folder '{folder_path}' dibuat.")
        # eval_env = RecordEpisodeStatistics(eval_env, buffer_length=1)  # log reward vektor/episode [file:22]
        # eval_env = RecordVideo(
        #             eval_env,
        #             video_folder=f"videos/{args.method}/{args.env_name}",
        #             episode_trigger=lambda ep: ep % 100 == 0,
        #         )

        agent.train(total_timesteps=args.ts, eval_env=eval_env, ref_point=ref_point)