"""Moderate State"""

from typing import Any

import awkward as ak
import numpy as np
import pandas as pd
import os,sys
import random
from random import randrange

import pandas as pd

sys.path.append("/home/rifnyrachman7/_merlion/ds-project-messiah/src")
from messiah.generators.base import Generator
from messiah.state import State

# Read Input file
file = "/home/rifnyrachman7/_merlion/data_input_v2.xlsx"
df = pd.read_excel(file, sheet_name="Parameters")

# Edge-level parameters
emission_edge = df["GHG_Unit"].iloc[:21].to_numpy()
emission_edge = np.concatenate([emission_edge, [0, 0, 0]])  # Supplying to 3 markets costs 0
cost_edge = df["Cost_process"].iloc[:21].to_numpy()
cost_edge = np.concatenate([cost_edge, [0, 0, 0]])  # Add 0 values at 3 markets

# Node-level parameters
emission_node = df["GHG_Unit"].iloc[21:].to_numpy()
emission_node = np.concatenate([[0, 0], emission_node, [0, 0, 0]])  # Supplier + markets = 0
cost_node = df["Cost_Inv"].iloc[21:].to_numpy()
cost_node = np.concatenate([[0, 0], cost_node, [0, 0, 0]]) # Plus 2 suppliers and 3 markets

# Define initial inventories
init_inv_series = df["Initial_Inv"].iloc[21:].to_numpy()           # length 8
init_inv_series = np.nan_to_num(init_inv_series, nan=0.0)         # safety
init_inv_series = np.clip(init_inv_series, 0, None)

class ModerateState(Generator):
    
    def __init__(
        self,
        num_timesteps: int,
        datetime_freq:str = "d",
        datetime_start: pd.Timestamp = None,
        #randomise: bool = True, # Assuming all parameters are randomised
        **kwargs,
    ) -> None:
        
        super().__init__(num_timesteps, datetime_freq, datetime_start, **kwargs)
        
        #self.randomise = randomise

        #define constants
        self.num_timesteps = num_timesteps
        self.num_nodes = 13
        self.num_edges = 24
        self.num_products = 2
        self.num_costs = 2
        self.num_market = 3

        #define the names dictionary
        node_names = np.array(
            ["supplier0", "supplier1", "factory2", "factory3", "factory4", "warehouse5", "warehouse6",  "retailer7", "retailer8", "retailer9", "market10", "market11", "market12"],
            dtype=np.str_,
        )
        edge_names = np.array(
            [
                "supplier0_factory2", # Raw material supply
                "supplier0_factory3",
                "supplier0_factory4",
                "supplier1_factory2",
                "supplier1_factory3",
                "supplier1_factory4",
                
                "factory2_factory2", # Production
                "factory3_factory3",
                "factory4_factory4",
                
                "factory2_warehouse5", # Distribution
                "factory2_warehouse6",
                "factory3_warehouse5",
                "factory3_warehouse6",
                "factory4_warehouse5",
                "factory4_warehouse6",

                "warehouse5_retailer7", # Supply to retailer
                "warehouse5_retailer8",
                "warehouse5_retailer9",
                "warehouse6_retailer7",
                "warehouse6_retailer8",
                "warehouse6_retailer9",
                
                "retailer7_market10", # Demand fulfillment
                "retailer8_market11",
                "retailer9_market12",
            ],
            dtype=np.str_,
        )
        product_names = np.array(["raw", "finished"], dtype=np.str_)
        cost_names = np.array(["monetary","emission"],dtype=np.str_)
        node_tags = ak.Array(
            [
                ["supplier"],
                ["supplier"],
                ["factory"],
                ["factory"],
                ["factory"],
                ["warehouse"],
                ["warehouse"],
                ["retailer"],
                ["retailer"],
                ["retailer"],
                ["market"],
                ["market"],
                ["market"],
            ]
        )
        edge_tags = ak.Array(
            [
                ["supply"],
                ["supply"],
                ["supply"],
                ["supply"],
                ["supply"],
                ["supply"],
                ["production"],
                ["production"],
                ["production"],
                ["distribution"],
                ["distribution"],
                ["distribution"],
                ["distribution"],
                ["distribution"],
                ["distribution"],
                ["distribution"],
                ["distribution"],
                ["distribution"],
                ["distribution"],
                ["distribution"],
                ["distribution"],
                ["demand"],
                ["demand"],
                ["demand"],
            ]
        )
        product_tags = ak.Array(
            [
                ["raw"],
                ["finished"],
            ]
        )
        cost_tags = ak.Array(
            [
                ["monetary"],
                ["emission"],
            ]
        )
        
        self.names = {
            "node_names": node_names,
            "edge_names": edge_names,
            "cost_names": cost_names,
            "product_names": product_names,
            "node_tags": node_tags,
            "edge_tags": edge_tags,
            "product_tags": product_tags,
            "cost_tags": cost_tags,
        }

        #define the upstream and downstream nodes
        self.edge_upstream_nodes = np.array([0, 0, 0, 1, 1, 1, 2, 3, 4, 2, 2, 3, 3, 4, 4,
                                             5, 5, 5, 6, 6, 6, 7, 8, 9
                                            ], dtype=self.int_dtype)
        self.edge_downstream_nodes = np.array([ 2, 3, 4, 2, 3, 4, 2, 3, 4, 5, 6, 5, 6, 5, 6,
                                               7, 8, 9, 7, 8, 9, 10, 11, 12
                                              ],dtype=self.int_dtype)

        #define the input and output products based on the product_tags
        self.edge_input_products = np.array(
            [[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]],
            dtype=self.int_dtype,
        )

        self.edge_output_products = np.array(
            [[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]],
            dtype=self.int_dtype,
        )

        self.edge_min_multiples = np.array(
            [1 for _ in range(self.num_edges)], dtype=self.int_dtype # Assuming all MOQ is 1
        )

        # Let's have slightly more complex edge lengths (but no variability for now)
        self.edge_expected_lengths = np.array([self.random_param(2, 1) for _ in range(self.num_edges)], dtype=self.int_dtype) # Randomise lead time
        self.edge_actual_lengths = np.tile(
            self.edge_expected_lengths, (num_timesteps, 1)
        ).T  # Each edge is expected to take 1 timestep
        
        # === Initialise random edge costs ===
        # Shape: (num_edges, num_costs, num_timesteps)
        self.edge_costs = np.zeros(
            (self.num_edges, self.num_costs, self.num_timesteps),
            dtype=np.float32
        )

        # Monetary costs
        for i in range(self.num_edges): # Random factor is set to negative for costs
            random_factors = self.random_param(
                loc=1.0, scale=0.1, size=self.num_timesteps
            )
            self.edge_costs[i, 0, :] = -cost_edge[i] * random_factors

        # Emission costs with different random factors
        for i in range(self.num_edges):
            random_factors = self.random_param(
                loc=1.0, scale=0.1, size=self.num_timesteps
            )
            self.edge_costs[i, 1, :] = -emission_edge[i] * random_factors

        # Overwrite the last 2 edges (prices), shape = num_edge, num_cost, num_timesteps
        random_factors = self.random_param(
            loc=1.0, scale=0.1, size=self.num_timesteps
        )
        base_prices = np.array([20, 21, 20.5]) # num_cost is set only for monetary

        self.edge_costs[-self.num_market:, 0, :] = base_prices[:, np.newaxis] * random_factors[np.newaxis, :]
        
        # Start with 0 movements
        self.edge_inputs = np.zeros(
            (self.num_edges, self.num_timesteps), dtype=self.int_dtype
        )  # Start with 0 movements
        self.edge_outputs = np.zeros(
            (self.num_edges, self.num_timesteps), dtype=np.float32
        )  # Start with 0 movements
        self.node_control = np.array(
            [False, False, True, True, True, True, True, True, True, True, False, False, False], dtype=np.bool_
        )  # Only the factory and retailer controlled by us
        self.node_inventory = np.zeros(
            (self.num_nodes, self.num_products, self.num_timesteps), dtype=self.int_dtype
        )
        
        for i in range(8):
            self.node_inventory[i, 1, :] = init_inv_series[i]
        
        self.node_wastage = np.zeros(
            (self.num_nodes, self.num_products, self.num_timesteps), dtype=self.int_dtype
        )  # Start with 0 inventory, therefore no initial wastage
        self.product_shelf_lives = np.full(
            self.num_products, self.num_timesteps, dtype=self.int_dtype
        )  # Products last for the whole episode
        
        # === Initialise random node cost ===
        # Shape: (num_nodes * num_costs * num_timesteps) # Assume similar cost for raw/finished
        self.node_costs = np.zeros(
            (self.num_nodes, self.num_products, self.num_costs, self.num_timesteps), dtype=np.float32
        )  # Each product costs 1 to hold
        # Monetary node costs
        for i in range(self.num_nodes):
            random_factors = self.random_param(
                loc=1, scale=0.1, size=self.num_timesteps
            )
            self.node_costs[i, :, 0, :] = -cost_node[i] * random_factors
        # Emission node costs
        for i in range(self.num_nodes):
            random_factors = self.random_param(
                loc=1, scale=0.1, size=self.num_timesteps
            )
            self.node_costs[i, :, 1, :] = -emission_node[i] * random_factors
        
        self.cost_counts = np.zeros(
            (self.num_costs, self.num_timesteps), dtype=np.float32
        )  # Start with 0 costs
        self.edge_yields = np.full(
           (self.num_edges, self.num_timesteps), 1.0, dtype=np.float32
        )
        self.edge_orders = np.zeros(
            (self.num_edges, self.num_timesteps), dtype=self.int_dtype
        )
        
    # Define function to randomise parameters
    def define_demand(self) -> np.ndarray: # Generate random demand every episode
        
        random_dist = random.choice(['poisson','normal'])
        if random_dist == 'poisson':
            lam = randrange(100,200)
            self.edge_orders[-self.num_market:self.num_edges,:] = np.random.poisson(
                lam=lam, size=(self.num_market, self.num_timesteps))

        else:
            loc = randrange(100,150)
            scale = randrange(40,60)
            self.edge_orders[-self.num_market:self.num_edges,:] = np.random.normal(
                loc=loc, scale=scale, size=(self.num_market, self.num_timesteps))

        return self.edge_orders
    
    def random_param(self, loc=0, scale=0, size=None): # Generate random state every episode
        return np.random.normal(loc=loc, scale=scale, size=size) # Return a scalar value
    
    def get_config(self) -> dict: #additional function to allign with env_wrapper
        """Returns a JSON-serializable config of this generator."""
        return {
            "num_timesteps": self.num_timesteps,
            "datetime_freq": self.datetime_freq,
            "datetime_start": self.datetime_start.isoformat() if self.datetime_start else None,
            #"randomise": self.randomise,
        }
    
    def __call__(self) -> tuple[State, dict[str, Any] | None]:
        """ Randomise the state when the new episode is called """
        self.edge_inputs.fill(0)
        self.edge_outputs.fill(0)
        self.node_wastage.fill(0)
        self.cost_counts.fill(0)
        self.edge_orders.fill(0)
        self.node_inventory.fill(0) # Similar to init, no need redefining
        
        for i in range(8):
            self.node_inventory[i, 1, :] = init_inv_series[i]
        
        self.define_demand() # Randomise demand every state generation
        
        # === Randomise edge parameters every episode ===
        # Shape: (num_edges, num_costs, num_timesteps)
        self.edge_costs.fill(0)

        # Monetary costs
        for i in range(self.num_edges):
            random_factors = self.random_param(
                loc=1.0, scale=0.1, size=self.num_timesteps
            )
            self.edge_costs[i, 0, :] = -cost_edge[i] * random_factors

        # Emission costs with different random factors
            random_factors = self.random_param(
                loc=1.0, scale=0.1, size=self.num_timesteps
            )
            self.edge_costs[i, 1, :] = -emission_edge[i] * random_factors

        # Overwrite the last 2 edges (prices), shape = num_edge, num_cost, num_timesteps
        random_factors = self.random_param(
            loc=1.0, scale=0.1, size=self.num_timesteps
        )
        base_prices = np.array([20, 21, 20.5]) # num_cost is set only for monetary

        self.edge_costs[-self.num_market:, 0, :] = base_prices[:, np.newaxis] * random_factors[np.newaxis, :]
        
        # === Randomise nodes parameters every episode ===
        self.node_costs.fill(0)
        
        for i in range(self.num_nodes):
            random_factors = self.random_param( # Randomise the node costs
                loc=1.0, scale=0.1, size=self.num_timesteps
            )
            self.node_costs[i, :, 0, :] = -cost_node[i] * random_factors
            
            random_factors = self.random_param( # Randomise the node emissions
                loc=1.0, scale=0.1, size=self.num_timesteps
            )
            self.node_costs[i, :, 1, :] = -emission_node[i] * random_factors
            
        
        return State(
            num_timesteps=self.num_timesteps,
            dates=self.dates,
            names=self.names,
            edge_upstream_nodes=self.edge_upstream_nodes,
            edge_downstream_nodes=self.edge_downstream_nodes,
            edge_input_products=self.edge_input_products,
            edge_output_products=self.edge_output_products,
            edge_min_multiples=self.edge_min_multiples,
            edge_yields=self.edge_yields,
            edge_expected_lengths=self.edge_expected_lengths,
            edge_actual_lengths=self.edge_actual_lengths,
            edge_costs=self.edge_costs,
            edge_orders=self.edge_orders,
            edge_inputs=self.edge_inputs,
            edge_outputs=self.edge_outputs,
            node_control=self.node_control,
            node_inventory=self.node_inventory,
            node_wastage=self.node_wastage,
            product_shelf_lives=self.product_shelf_lives,
            node_costs=self.node_costs,
            cost_counts=self.cost_counts,
            include_wastage=True,
        ), None
