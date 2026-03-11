import mo_gymnasium as mo_gym

import numpy as np

from morl_baselines.multi_policy.multi_policy_moqlearning.mp_mo_q_learning import MPMOQLearning
from morl_baselines.multi_policy.morld.morld import MORLD
from morl_baselines.multi_policy.pgmorl.pgmorl import PGMORL
from morl_baselines.multi_policy.capql.capql import CAPQL
from morl_baselines.multi_policy.pareto_q_learning.pql import PQL
from morl_baselines.multi_policy.pcn.pcn import PCN
from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPIPD
from morl_baselines.multi_policy.gpi_pd.gpi_pd_continuous_action import GPIPDContinuousAction

from gymnasium.wrappers import TimeLimit
from gymnasium.wrappers import FlattenObservation, RecordVideo
# from mo_gymnasium.wrappers import RecordEpisodeStatistics
# from mo_gymnasium.wrappers.vector import MORecordEpisodeStatistics
# from mo_gymnasium.wrappers.wrappers import MORecordEpisodeStatistics
# from mo_gymnasium.utils import MORecordEpisodeStatistics
# from gymnasium.wrappers import RecordEpisodeStatistics


# from morl_baselines.multi_policy.linear_support.linear_support import OLS

ENV_NAME = [
    "mo-halfcheetah-v4",
    "deep-sea-treasure-v0",
    "resource-gathering-v0",
    "mo-mountaincar-v0",
    "mo-mountaincarcontinuous-v0",
    "mo-lunar-lander-v2",
    "water-reservoir-v0",
    "four-room-v0",
    "mo-highway-fast-v0",
    "mo-reacher-v4",
    "mo-hopper-v4"
]

OBS_SPACE = {
    "mo-halfcheetah-v4": "continuous",         # mujoco
    "deep-sea-treasure-v0": "discrete",
    "resource-gathering-v0": "discrete",
    "mo-mountaincar-v0": "continuous",         # asli diskrit, MO: continuous obs
    "mo-mountaincarcontinuous-v0": "continuous",
    "mo-lunar-lander-v2": "continuous",
    "water-reservoir-v0": "continuous",
    "four-room-v0": "discrete",
    "mo-highway-fast-v0": "continuous",
    "mo-reacher-v4": "continuous",             # mujoco
    "mo-hopper-v4": "continuous"               # mujoco
}

ACTION_SPACE = {
    "mo-halfcheetah-v4": "continuous",         # mujoco
    "deep-sea-treasure-v0": "discrete",
    "resource-gathering-v0": "discrete",
    "mo-mountaincar-v0": "discrete",
    "mo-mountaincarcontinuous-v0": "continuous",
    "mo-lunar-lander-v2": "discrete",  # tergantung varian, yang continuous pakai yang ada -continuousnya
    "water-reservoir-v0": "continuous",
    "four-room-v0": "discrete",
    "mo-highway-fast-v0": "discrete",
    "mo-reacher-v4": "discrete",
    "mo-hopper-v4": "continuous"
}

NUM_OBJ = {
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

## rollout_fragment_length
MAX_EPISODE = {
    "mo-halfcheetah-v4": 1000,           # locomotion panjang, standar MuJoCo [web:217]
    "deep-sea-treasure-v0": 200,         # grid kecil, treasure tercapai <~30 langkah [web:180]
    "resource-gathering-v0": 200,        # grid kecil, episode relatif pendek [web:191]
    "mo-mountaincar-v0": 1000,           # default max_episode_steps=200 [web:201][web:202]
    "mo-mountaincarcontinuous-v0": 1000, # default max_episode_steps=999 [web:210]
    "mo-lunar-lander-v2": 1000,          # max_episode_steps=1000 [web:106]
    "water-reservoir-v0": 200,           # horizon operasional moderat [web:233]
    "four-room-v0": 400,                 # grid 4-room, perlu horizon agak panjang [web:186]
    "mo-highway-fast-v0": 200,           # duration ~40 step per episode [web:236]
    "mo-reacher-v4": 200,                # max_episode_steps=50 [web:222]
    "mo-hopper-v4": 1000,                # locomotion panjang, standar MuJoCo [web:217]
}

## reference point for reward.
REF_POINT = {
    "mo-halfcheetah-v4": np.array([-100.0, -100.0]), ### follow morl_baselines          # locomotion panjang, standar MuJoCo [web:217]
    "deep-sea-treasure-v0": np.array([1., -0.9]),               # grid kecil, treasure tercapai <~30 langkah [web:180]
    "resource-gathering-v0": np.array([-0.99, 0.01, 0.01]),     # grid kecil, episode relatif pendek [web:191]
    "mo-mountaincar-v0": np.array([-1., -0.99, -0.99]),         # default max_episode_steps=200 [web:201][web:202]
    "mo-mountaincarcontinuous-v0": np.array([-0.99, -0.99]),    # default max_episode_steps=999 [web:210]
    "mo-lunar-lander-v2": np.array([-101., -1001, -101., -101.]), # follow morl_baselines, max_episode_steps=1000 [web:106]
    "water-reservoir-v0": np.array([-999999, -999999]),         # horizon operasional moderat [web:233]
    "four-room-v0": np.array([0.01, 0.01, 0.01]),               # grid 4-room, perlu horizon agak panjang [web:186]
    "mo-highway-fast-v0": np.array([0.01, 0.01, -0.99]),        # duration ~40 step per episode [web:236]
    "mo-reacher-v4": np.array([-0.99, -0.99, -0.99, -0.99]),    # max_episode_steps=50 [web:222]
    "mo-hopper-v4": np.array([-100, -100, -100]),      # follow morl_baselines, locomotion panjang, standar MuJoCo [web:217]
}

def get_algorithms(env_name: str = 'deep-sea-treasure-v0', name:str = 'morld', seed = 42):
    env = mo_gym.make(env_name, render_mode="rgb_array")
    if 'mo-highway' in env_name:
        env = FlattenObservation(env)
    env = TimeLimit(env, max_episode_steps=MAX_EPISODE[env_name])
        
    # env = RecordEpisodeStatistics(env, buffer_length=1)
    # env = MORecordEpisodeStatistics(env, )
    ref_point = REF_POINT[env_name]
    # ref_point = np.array([0.0]*NUM_OBJ[env_name])
    assert name in ['morld', 'pgmorl', 'capql', 'pql', 'mpmoql', 'pcn'], f"environment name should be between {['morld', 'pgmorl', 'capql', 'pql', 'mpmoql', 'pcn']}"
    if name == 'morld':
        policy_nm = 'MOSAC' if ACTION_SPACE[env_name] == 'continuous' else 'MOSACDiscrete'
        # policy_dict = {'net_arch': [256, 256], 'policy_lr': 1e-2, 'q_lr': 1e-3} if ACTION_SPACE[env_name] == 'continuous' else {'net_arch': [50], 'learning_rate': 1e-3}
        agent = MORLD(
        env,
        scalarization_method= "ws",  # "ws" or "tch"
        evaluation_mode = "ser",  # "esr" or "ser"
        policy_name = policy_nm,
        policy_args = {'net_arch': [64, 64], 'policy_lr': 1e-4, 'q_lr': 1e-2},
        # default policy_args = {'net_arch': [256, 256], 'policy_lr': 1e-2, 'q_lr': 1e-1},
        gamma = 0.99, ## default was 0.995
        pop_size = 10, ## default was 6
        seed = seed, ## default 42
        exchange_every = int(4e4),
        neighborhood_size = 1,  # n = "n closest neighbors", 0=none
        dist_metric = lambda a, b: np.sum(
            np.square(a - b)
        ),  # distance metric between neighbors
        project_name= "MORL-Baselines",
        experiment_name= "MORL-D" if seed is None else f'MORL-D-{seed}',
        wandb_entity = 'merlion-project',
        shared_buffer = False,
        update_passes = 10,
        weight_init_method = "uniform",
        log = True,
    )
    elif name == 'pgmorl':
        """Prediction Guided Multi-Objective Reinforcement Learning.

        Reference: J. Xu, Y. Tian, P. Ma, D. Rus, S. Sueda, and W. Matusik,
        “Prediction-Guided Multi-Objective Reinforcement Learning for Continuous Robot Control,”
        in Proceedings of the 37th International Conference on Machine Learning,
        Nov. 2020, pp. 10607–10616. Available: https://proceedings.mlr.press/v119/xu20h.html

        Paper: https://people.csail.mit.edu/jiex/papers/PGMORL/paper.pdf
        Supplementary materials: https://people.csail.mit.edu/jiex/papers/PGMORL/supp.pdf
        """
        ## the id is a string, not gym make
        ## origin (reference point) is here as well instead of in train. need to do it twice
        assert OBS_SPACE[env_name] == 'continuous' and ACTION_SPACE[env_name] == 'continuous', f'Both observation and action space of problem should be continuous. Your problem {env_name} has {OBS_SPACE[env_name]} observation space and {ACTION_SPACE[env_name]} action space'
        agent = PGMORL(
            env_id = env_name, 
            origin = ref_point, 
            num_envs = 4,
            pop_size = 10, ## default was 6
            warmup_iterations = int(MAX_EPISODE[env_name] * 0.05), ## default 80
            steps_per_iteration = MAX_EPISODE[env_name], ## default was 2048
            evolutionary_iterations = 20,
            num_weight_candidates = 21 if NUM_OBJ[env_name] == 2 else 32, ## default was 7
            num_performance_buffer = 100,
            performance_buffer_size = 2,
            min_weight = 0.0,
            max_weight = 1.0,
            delta_weight = 1/9, ### npop=1/delta weight + 1, default 0.2
            sparsity_coef = -1.0,
            env=None,
            gamma = 0.99,
            project_name = "MORL-baselines",
            experiment_name = "PGMORL" if seed is None else f'PGMORL-{seed}',
            wandb_entity = 'merlion-project',
            seed = seed,
            log = True,
            net_arch = [64, 64],
            num_minibatches = 32,
            update_epochs = 10,
            learning_rate = 3e-4,
            anneal_lr = False,
            clip_coef = 0.2,
            ent_coef = 0.0,
            vf_coef = 0.5,
            clip_vloss = True,
            max_grad_norm = 0.5,
            norm_adv = True,
            target_kl = None,
            gae = True,
            gae_lambda = 0.95,
            device = "auto",
            group = None,
        )
    elif name == 'capql':
        """CAPQL algorithm.

        MULTI-OBJECTIVE REINFORCEMENT LEARNING: CONVEXITY, STATIONARITY AND PARETO OPTIMALITY
        Haoye Lu, Daniel Herman & Yaoliang Yu
        ICLR 2023
        Paper: https://openreview.net/pdf?id=TjEzIsyEsQ6
        Code based on: https://github.com/haoyelu/CAPQL
        """
        assert OBS_SPACE[env_name] == 'continuous' and ACTION_SPACE[env_name] == 'continuous', f'Both observation and action space of problem should be continuous. Your problem {env_name} has {OBS_SPACE[env_name]} observation space and {ACTION_SPACE[env_name]} action space'
        agent = CAPQL(
            env,
            learning_rate = 3e-4,
            gamma = 0.99, ## default was 0.99
            tau = 0.005,
            buffer_size = 1000000,
            net_arch = [64, 64], ## [256, 256]
            batch_size = MAX_EPISODE[env_name], ## default 128
            num_q_nets = 2,
            alpha = 0.2,
            learning_starts = 1000,
            gradient_updates = 1,
            project_name = "MORL-Baselines",
            experiment_name = "CAPQL"  if seed is None else f'CAPQL-{seed}',
            wandb_entity = 'merlion-project',
            log = True,
            seed = seed,
            device = "auto",
        )
    elif name == 'pql':
        """Pareto Q-learning.
        Tabular method relying on pareto pruning.
        Paper: K. Van Moffaert and A. Nowé, “Multi-objective reinforcement learning using sets of pareto dominating policies,” The Journal of Machine Learning Research, vol. 15, no. 1, pp. 3483–3512, 2014.
        """
        assert OBS_SPACE[env_name] == 'discrete' and ACTION_SPACE[env_name] == 'discrete', f'Both observation and action space of problem should be discrete. Your problem {env_name} has {OBS_SPACE[env_name]} observation space and {ACTION_SPACE[env_name]} action space'

        agent = PQL(
            env,
            ref_point = ref_point,
            gamma = 0.99, ## default was 0.8
            initial_epsilon = 1.0,
            epsilon_decay_steps = 100000,
            final_epsilon = 0.1,
            seed = seed,
            project_name = "MORL-Baselines",
            experiment_name = "Pareto Q-Learning" if seed is None else f'Pareto Q-Learning-{seed}',
            wandb_entity = 'merlion-project',
            log = True,
        )   

    elif name == 'mpmoql':
        """Multi-policy MOQ-Learning: Outer loop version of mo_q_learning.

        Paper: Paper: K. Van Moffaert, M. Drugan, and A. Nowe, Scalarized Multi-Objective Reinforcement Learning: Novel Design Techniques. 2013. doi: 10.1109/ADPRL.2013.6615007.
        """

        assert OBS_SPACE[env_name] == 'discrete' and ACTION_SPACE[env_name] == 'discrete', f'Both observation and action space of problem should be discrete. Your problem {env_name} has {OBS_SPACE[env_name]} observation space and {ACTION_SPACE[env_name]} action space'
        agent = MPMOQLearning(
            env,
            # scalarization=weighted_sum, # default is weighted_sum
            learning_rate = 0.1,
            gamma = 0.99, ## default was 0.9
            initial_epsilon = 0.1,
            final_epsilon = 0.1,
            epsilon_decay_steps = None, ## default was 100000
            weight_selection_algo = "random",
            epsilon_ols = None,
            use_gpi_policy = False,
            transfer_q_table = True,
            dyna = False,
            dyna_updates = 5,
            gpi_pd = False,
            project_name = "MORL-Baselines",
            experiment_name = "MultiPolicy MO Q-Learning"  if seed is None else f'MultiPolicy MO Q-Learning-{seed}',
            wandb_entity = 'merlion-project',
            seed = seed,
            log = True,
        )
    elif name == 'pcn':
        """Pareto Conditioned Networks (PCN).

        Reymond, M., Bargiacchi, E., & Nowé, A. (2022, May). Pareto Conditioned Networks.
        In Proceedings of the 21st International Conference on Autonomous Agents
        and Multiagent Systems (pp. 1110-1118).
        https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p1110.pdf

        ## Credits

        This code is a refactor of the code from the authors of the paper, available at:
        https://github.com/mathieu-reymond/pareto-conditioned-networks
        """
        assert OBS_SPACE[env_name] == 'continuous', f'Observation space should be continuous. Your problem {env_name} has {OBS_SPACE[env_name]} observation space'
        
        agent = PCN(
            env,
            scaling_factor = np.array([1.0]),
            learning_rate = 1e-3,
            gamma = 0.99, ## default was 1.0
            batch_size = MAX_EPISODE[env_name],
            hidden_dim = 64,
            noise = 0.1,
            project_name = "MORL-Baselines",
            experiment_name = "PCN" if seed is None else f'PCN-{seed}',
            wandb_entity = 'merlion-project',
            log = True,
            seed = seed,
            device = "auto",
            # model_class: Optional[Type[BasePCNModel]] = None,
        )

        agent.continuous_action = ACTION_SPACE[env_name] == 'continuous'
    # elif name == 'ols':
    #     agent = OLS(
    #         num_objectives,
    #         epsilon: float = 0.0,
    #         verbose: bool = True,
    #     )
    else:
        raise Exception(f'need to add in benchmark_algorithms_call.py: {name}')
    return agent, env, ref_point