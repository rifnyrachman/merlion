import logging
import time
import warnings
from dataclasses import make_dataclass

import numpy as np
import gymnasium as gym
import random
import pickle

import sys
sys.path.append("/home/rifnyrachman7/_merlion") #access to this folder
sys.path.append("/home/rifnyrachman7/_merlion/ds-project-messiah/src/messiah") #access to messiah
from state_generator.moderate_state import ModerateState
from gymnasium import spaces

from messiah.agents.base import Agent
from messiah.generators.base import Generator
from messiah.history.base import History
from messiah.ops import count_node_costs, remove_wasted_inventory, start_processes
from messiah.agents.processing import MaxAvailableProcessingAgent, MaxOrderedProcessingAgent, FixedProcessingAgent
from messiah.agents.utils import allocate_greedily


logger = logging.getLogger("messiah")
logger.setLevel("DEBUG")

#set objective value range
#set reward range
# min_profit = 190.22
# max_profit = 5679.69
min_profit = 0
max_profit = 5000 # previously 2000
min_emission = 0 # previously 7.38
max_emission = 2500 # previously 2500 # previously 3596.1
max_equity = 2 #previously 2
min_equity = 0

#define the agents
agents = [
    MaxAvailableProcessingAgent(edge_tags = ["production"], max_multiple = 1),
    MaxAvailableProcessingAgent(edge_tags = ["distribution"], max_multiple = 1),
    MaxOrderedProcessingAgent(edge_tags = ["demand"], max_multiple = 1, part_fulfil = True),
    FixedProcessingAgent(multiple = 1, edge_tags = ["supply"], part_fulfil = True)
]

# [NEW] thin wrapper so RLlib/VectorEnv wrappers can still call into env
class ScalarizationWeightProxy(gym.Wrapper):
    """Forward (get|set)_scalarization_weights through any wrapper chain."""
    def set_scalarization_weights(self, w):
        # Forward to the base env if the wrapper doesn’t implement it
        if hasattr(self.env, "set_scalarization_weights"):
            return self.env.set_scalarization_weights(w)
        raise AttributeError("Underlying env has no set_scalarization_weights(w)")

    def get_scalarization_weights(self):
        if hasattr(self.env, "get_scalarization_weights"):
            return self.env.get_scalarization_weights()
        return None

class ModerateSC(gym.Env):
    def __init__(self, *args, **kwargs):
        self.agents = agents
        self.artefacts = None
        
        self.emission = 0
        self.inequality = 0
        
        self.cum_cost_per_edge = 0
        self.cum_emission_per_edge = 0
        self.cum_emission = 0
        self.average_ineq = 0
        
        self.vector_reward = None
        self.scalar_reward = 0
        self.timestep = 0
        
        self.max_flow = 200
        self.max_inv = 10_000
        
        # Initiate costs
        self.step_edge_cost     = 0
        self.step_edge_emission = 0
        self.step_node_cost     = 0
        self.step_node_emission = 0
        self.total_cost = 0
        self.total_emission = 0
        
        self.emission = 0.0
        self.inequality = 0.0
        
        self.set_task() # Call initial state
        self.agent_times = {agent: 0.0 for agent in self.agents}
        
        self.max_timestep = self.state.num_timesteps
        
        obs_shape = self._get_obs().shape
        self.observation_space = spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)

        action_shape = (self.state.num_edges,)
        self.action_space = spaces.Box(low=-1, high=1, shape=action_shape, dtype=np.float32)
        self.reward_to_save = []
        self.save_filename = None
        
        self.num_objectives = 3
        self._w = None
        
    # --- NEW: external control of scalarization weights ---
    def set_scalarization_weights(self, w):
        w = np.asarray(w, dtype=np.float32).reshape(-1)
        if w.size != self.num_objectives:
            raise ValueError(f"Expected {self.num_objectives} weights, got {w.size}")
        w = np.clip(w, 1e-12, None)
        self._w = w / w.sum()

    def get_scalarization_weights(self):
        return None if self._w is None else self._w.copy()
        
    def _get_obs(self):    
        
        # inv = np.zeros(
        #     ((self.state.num_nodes-2), self.state.num_products)
        # ) # Assuming 0 initial inventory
        # flow = np.zeros((self.state.num_edges), dtype=np.float32)
        
        inv = self.state.node_inventory[:, :, self.timestep]/self.max_inv # Call inventory at each time step
        flow = self.state.edge_outputs[:, self.timestep]/self.max_flow # Call order at each time step
        # TODO: Add cumulative emission and inequality
        arr_obs = np.concatenate((
            inv.flatten(),flow.flatten(),
            np.array([self.emission, self.inequality], dtype=np.float32)
                                 ))
        
        emission = min(max((self.emission/1e6)/1e6, 0), 1) #first 1e6 for decimal values, second for normalisation
        inequality = min(max((self.inequality)/(self.timestep+1), 0), 1)

        arr_obs_final = np.concatenate((arr_obs,[emission,inequality]))
        arr_obs_final = np.clip(arr_obs_final, 0, 1) #clip obs within abs space
        arr_obs_final = np.clip([_ for _ in arr_obs], 0.0, 1.0).astype(np.float32)
        return arr_obs_final
        
        #TODO: normalise the inventory and flow, add cumulative emission and average inequality
        
    def _calculate_cost(self, t: int, multiples: np.ndarray):
        # --- edge costs at this timestep (charged on start) ---
        step_edge_cost     = float(np.sum(self.state.edge_costs[:, 0, t] * multiples))
        step_edge_emission = float(np.sum(self.state.edge_costs[:, 1, t] * multiples))

        # --- node holding costs/emissions at this timestep via delta ---
        prev_monetary  = float(self.state.cost_counts[0, t])
        prev_emission  = float(self.state.cost_counts[1, t])
        
        # print(
        #     "prev_monetary", prev_monetary,
        #     "prev_emission", prev_emission
        # )

        # Mutates cost_counts[:, t] in-place by ADDING step holding costs
        count_node_costs(
            t,
            self.state.node_costs,
            self.state.node_inventory,
            self.state.node_control,
            self.state.cost_counts,
        ) # TODO: make sure this is called properly. It doesn't print
        
        step_node_cost     = float(self.state.cost_counts[0, t] - prev_monetary)
        step_node_emission = float(self.state.cost_counts[1, t] - prev_emission)
        
        # print(
        #     "self.state.cost_counts[0, t]:", self.state.cost_counts[0, t],
        #     "float(self.state.cost_counts[1, t]", float(self.state.cost_counts[1, t]
        #                                                )
        # )

        # step totals (this timestep only)
        self.step_cost     = step_edge_cost + step_node_cost
        self.step_emission = step_edge_emission + step_node_emission

        # (optional) maintain running totals for logging
        self.total_cost     += self.step_cost
        self.total_emission += self.step_emission

        # print(
        #     "timestep:", self.timestep,
        #     "step edge cost:", step_edge_cost,
        #     "step edge emission:", step_edge_emission,
        #     "step node cost:", step_node_cost,
        #     "step node emission:", step_node_emission
        # ) ## Debugging
        # return STEP values (use these for reward), plus totals if you want
        return self.step_cost, self.step_emission, self.total_cost, self.total_emission
    
    def _step_state(self):
        #print(f"entering _step_state() timestep-,{self.timestep}") ##debugging
        
        for agent in self.agents:
            #print(f"debug calling agent.apply for {agent}") ##debugging
            agent.apply(self.timestep, self.state, self.artefacts) #apply transform at each timestep
        
    def set_task(self):
        self.state = ModerateState(num_timesteps=100)()[0] #task
        self.state.include_wastage = False
        self._setup_agents_and_artefacts()
        
        # Precompute the agent apply plan given the generator we are using
        # iterative_agents = [a for a in self.agents if a.run_every != -1]
        # self.agent_apply_plan = {
        #     t: [a for a in iterative_agents if a.is_active(t)]
        #     for t in range(self.state.num_timesteps)
        # }
                                             
    def get_task(self):
        return self.state
                                
    def sample_tasks(self, n_tasks):
        #return n_tasks of new tasks/states
        return [ModerateState(num_timesteps=100)()[0] for _ in range(n_tasks)]
                                             
    # def set_weights(self): # weights are set in the MERLIONArchive
    #     self.weights = np.array(np.random.dirichlet(alpha=[1,1,1], size=1)[0])
    #     #print('weights:',self.weights) #for checking only
    #     return self.weights              
    
    def reset(self, *, seed=None, options=None):
        """Reset the environment.

        Parameters
        ----------
        start_timestep: int
            The timestep to start the environment at
        """
        super().reset(seed=seed)
        self.state = ModerateState(num_timesteps = 100)()[0] #generate a new state
        #self.weights = self.set_weights() #debugging
        if self._w is None:
            # default to uniform if trainer hasn’t pushed weights yet
            self.set_scalarization_weights(np.ones(self.num_objectives, dtype=np.float32))
        
        self.timestep = 0  # Reset the timestep counter
        self.cum_cost_per_edge = 0 #restart calculation of cumulative cost at edge
        self.cum_emission_per_edge = 0
        self.cum_emission = 0
        self.average_ineq = 0
        self.max_timestep = self.state.num_timesteps
        self.agent_times = {agent: 0.0 for agent in self.agents}
        
        self.vector_reward = np.array([0.0, 0.0, 0.0]) ##debugging
        self.scalar_reward = 0.0
        
        self.emission = 0.0
        self.inequality = 0.0
        
        # Initialize costs
        self.step_edge_cost = 0
        self.step_edge_emission = 0
        self.step_node_cost = 0
        self.step_node_emission = 0
        self.total_cost = 0
        self.total_emission = 0
        
        #Added: clear last multi reward tracker
        self.last_multi_reward = None
        
        self._setup_agents_and_artefacts()
                                             
        observation = self._get_obs().astype(np.float32)
        
        return observation, {}
    
    def _setup_agents_and_artefacts(self):
        # Create a dictionary to store the artefacts
        artefacts_data = {}
        # Setup the agents and collect their artefacts
        for agent in self.agents:
            agent_artefacts = agent.setup(self.state)
            if agent_artefacts:
                for key in agent_artefacts:
                    if key in artefacts_data:
                        msg = f"Duplicate key found in artefacts_data: {key}"
                        raise KeyError(msg)
                artefacts_data.update(agent_artefacts)

        if not artefacts_data:
            self.artefacts = None
        else:
            fields = [(key, type(value)) for key, value in artefacts_data.items()]
            artefacts_class = make_dataclass("DataClass", fields)
            self.artefacts = artefacts_class(**artefacts_data)
                                             
    def step(self, action):
        self._step_state() # RL agent learns delta policy rather than from scratch
        max_multiple = 200  # Max delivery quantity
        # Scale from (-1, 1) → (0, 1), then → (0, max_multiple), then round
        multiples = ((action + 1) / 2) * max_multiple
        multiples = np.round(multiples).astype(self.state.int_dtype)
        
        try:
            #print(f"multiple-{self.timestep}",multiples) ##debugging
            multiples = allocate_greedily( ##debugging from no 'self.state ='
                t=self.timestep,
                state=self.state,
                requested_multiples=multiples)

        except Exception as e:
            #print(self.state)
            print("start_processes failed:", e)
            # You may want to return zero reward or handle failure here
        
        start_processes(self.timestep, self.state, multiples, safe=True)

        """Run a single timestep of the environment."""
        if self.state.include_wastage:
            remove_wasted_inventory(
                self.timestep, self.state.node_inventory, self.state.node_wastage
            )

        if self._w is None:
            # be defensive; should have been set in reset() or by trainer
            self.set_scalarization_weights(np.ones(self.num_objectives, dtype=np.float32))
        self.scalar_reward = float(np.dot(self.vector_reward, self._w))
        
        terminated = False
        truncated = self.timestep >= self.max_timestep - 1
        
        # === Reward Calculation ===
        step_cost, step_emission, _, _ = self._calculate_cost(self.timestep, multiples)
        
        # Read demand and calculate service level
        demand = self.state.edge_orders[-3:self.state.num_edges, self.timestep]
        
        sl = np.zeros(3)
        epsilon = 1e-12
        sl[0] = min((self.state.edge_inputs[15, self.timestep]
                      + self.state.edge_inputs[18, self.timestep])/(demand[0]+epsilon), 1)
        sl[1] = min((self.state.edge_inputs[16, self.timestep]
                      + self.state.edge_inputs[19, self.timestep])/(demand[1]+epsilon), 1)
        sl[2] = min((self.state.edge_inputs[17, self.timestep]
                      + self.state.edge_inputs[20, self.timestep])/(demand[2]+epsilon), 1)
                                             
        inequality = 0
        for i in range(3):
            for j in range(3):
                if (j != i):
                    inequality += abs(sl[i]-sl[j])

        # Raw rewards (before scaling)
        reward_1 = step_cost # clip to prevent negative profit
        reward_2 = step_emission  # step_emission is already negative from moderate_state.py
        reward_3 = -0.5 * inequality   # Match exist_sim_param_state.py: -0.5 factor
        
        #print("inequality:",inequality,"reward 3 raw:", reward_3)

        # === Fixed Scaling ===
        # Ensure profit scaling handles negative values correctly
        # if raw_profit < min_profit:
        #     scaled_reward_1 = 0.0
        # elif raw_profit > max_profit:
        #     scaled_reward_1 = 1.0
        # else:
        scaled_reward_1 = (reward_1 - min_profit) / (max_profit - min_profit)

        # Emission scaling - match exist_sim_param_state.py: (max_emission + reward_2) / (max_emission - min_emission)
        # where reward_2 = step_emission (already negative)
        scaled_reward_2 = (max_emission + reward_2) / (max_emission - min_emission)

        # Inequality scaling - match exist_sim_param_state.py: max_equity + reward_3
        # where max_equity = 1 and reward_3 = -0.5 * inequality
        #scaled_reward_3 = max_equity + reward_3  # This gives: 1 + (-0.5 * inequality)
        scaled_reward_3 = (max_equity + reward_3) / (max_equity - min_equity)
        
        #print("scaled reward 3:", scaled_reward_3)
        
        self.reward_to_save.append([self.timestep, reward_1, reward_2, reward_3])
        # if terminated and self.save_filename is not None:
        #     with open(self.save_filename, 'wb') as f:
        #         pickle.dump(self.reward_to_save, f)
        
        # Create vector reward
        self.vector_reward = np.array([scaled_reward_1, scaled_reward_2, scaled_reward_3])
        #print(f"vector rewards:{self.vector_reward}") ##debugging

        # Ensure all rewards are in [0, 1]
        #self.vector_reward = np.clip(self.vector_reward, 0.0, 1.0)

        # Calculate scalar reward
        self.scalar_reward = np.dot(self.vector_reward, self._w)
        
        raw_emission = (max_emission - min_emission) * self.vector_reward[1] - max_emission

        # Accumulate raw emission over time
        self.emission += raw_emission

        # Update cumulative metrics for observation
        #self.emission += (((max_emission-min_emission)*self.vector_reward[1])-max_emission)
        self.inequality = ((self.inequality * self.timestep) + self.vector_reward[2]) / (self.timestep + 1)

        observation = self._get_obs()
        self.timestep += 1
        
        info = {
            "mo_reward": self.vector_reward.astype(np.float32),
            "mo_reward_raw": np.array([reward_1, reward_2, reward_3], dtype=np.float32),
            "weights": self._w
        }
        
        #print(info) ## debugging
        
        self.last_multi_reward = info["mo_reward"]

        # if self.timestep < 3:  # Debugging
        #     print(dict(
        #         t=self.timestep,
        #         rewards=[reward_1, reward_2, reward_3],
        #         scaled_rewards=self.vector_reward.tolist(),
        #         weights=self.weights.tolist(),
        #         scalar_reward=float(self.scalar_reward),
        #         step_cost=step_cost,
        #         step_emission=step_emission,
        #         inequality=inequality
        #     ))

        return observation, self.scalar_reward, truncated, False, info

def make_env(env_config=None):
    base = ModerateSC()
    return ScalarizationWeightProxy(base)
