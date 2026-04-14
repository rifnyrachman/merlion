"""Complex State"""

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
df = pd.read_excel(file, sheet_name="Parameters_complex")

# Edge-level parameters
emission_edge = df["GHG_Unit"].iloc[:59].to_numpy()
emission_edge = np.concatenate([emission_edge, [0, 0, 0, 0, 0]])  # Supplying to 3 markets costs 0
cost_edge = df["Cost_process"].iloc[:59].to_numpy()
cost_edge = np.concatenate([cost_edge, [0, 0, 0, 0, 0]])  # Add 0 values at 3 markets

# Node-level parameters
emission_node = df["GHG_Unit"].iloc[59:].to_numpy()
emission_node = np.concatenate([[0, 0, 0], emission_node, [0, 0, 0, 0, 0]])  # Supplier + markets = 0
cost_node = df["Cost_Inv"].iloc[59:].to_numpy()
cost_node = np.concatenate([[0, 0, 0], cost_node, [0, 0, 0, 0, 0]]) # Plus 2 suppliers and 3 markets

# Define initial inventories
init_inv_series = df["Initial_Inv"].iloc[59:].to_numpy()           # length 8
init_inv_series = np.nan_to_num(init_inv_series, nan=0.0)         # safety
init_inv_series = np.clip(init_inv_series, 0, None)

class ComplexState(Generator):
    
    def __init__(
        self,
        num_timesteps: int,
        datetime_freq:str = "d",
        datetime_start: pd.Timestamp = None,
        **kwargs,
    ) -> None:
        
        super().__init__(num_timesteps, datetime_freq, datetime_start, **kwargs)

        #define constants
        self.num_timesteps = num_timesteps
        self.num_nodes = 24
        self.num_edges = 64
        self.num_products = 2
        self.num_costs = 2
        self.num_market = 5

        #define the names dictionary
        node_names = np.array(
            ["supplier0", "supplier1", "supplier2", "factory3", "factory4", "factory5",
             "factory6", "factory7", "warehouse8", "warehouse9", "warehouse10", "distributor11",
             "distributor12", "distributor13", "retailer14", "retailer15", "retailer16",
             "retailer17", "retailer18", "market19", "market20", "market21", "market22", "market23"
            ],
            dtype=np.str_,
        )
        edge_names = np.array(
            [
                "supplier0_factory3",
                "supplier0_factory4",
                "supplier0_factory5",
                "supplier0_factory6",
                "supplier0_factory7",
                
                "supplier1_factory3",
                "supplier1_factory4",
                "supplier1_factory5",
                "supplier1_factory6",
                "supplier1_factory7",
                
                "supplier2_factory3",
                "supplier2_factory4",
                "supplier2_factory5",
                "supplier2_factory6",
                "supplier2_factory7",
                
                "factory3_warehouse8",
                "factory3_warehouse9",
                "factory3_warehouse10",
                
                "factory4_warehouse8",
                "factory4_warehouse9",
                "factory4_warehouse10",
                
                "factory5_warehouse8",
                "factory5_warehouse9",
                "factory5_warehouse10",
                
                "warehouse8_distributor11",
                "warehouse8_distributor12",
                "warehouse8_distributor13",
                
                "warehouse9_distributor11",
                "warehouse9_distributor12",
                "warehouse9_distributor13",
            
                "warehouse10_distributor11",
                "warehouse10_distributor12",
                "warehouse10_distributor13",
                
                "distributor11_retailer14",
                "distributor11_retailer15",
                "distributor11_retailer16",
                "distributor11_retailer17",
                "distributor11_retailer18",
                
                "distributor12_retailer14",
                "distributor12_retailer15",
                "distributor12_retailer16",
                "distributor12_retailer17",
                "distributor12_retailer18",
                
                "distributor13_retailer14",
                "distributor13_retailer15",
                "distributor13_retailer16",
                "distributor13_retailer17",
                "distributor13_retailer18",
                
                "retailer14_market19",
                "retailer15_market20",
                "retailer16_market21",
                "retailer17_market22",
                "retailer18_market23",
            ],
            dtype=np.str_,
        )
        product_names = np.array(["raw", "finished"], dtype=np.str_)
        cost_names = np.array(["monetary","emission"],dtype=np.str_)
        node_tags = ak.Array(
            [
                ["supplier"],
                ["supplier"],
                ["supplier"],
                ["factory"],
                ["factory"],
                ["factory"],
                ["factory"],
                ["factory"],
                ["warehouse"],
                ["warehouse"],
                ["warehouse"],
                ["distributor"],
                ["distributor"],
                ["distributor"],
                ["retailer"],
                ["retailer"],
                ["retailer"],
                ["retailer"],
                ["retailer"],
                ["market"],
                ["market"],
                ["market"],
                ["market"],
                ["market"]
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
                ["supply"],
                ["supply"],
                ["supply"],
                ["supply"],
                ["supply"],
                ["supply"],
                ["supply"],
                ["supply"],
                ["supply"],
                ["production"],
                ["production"],
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
                ["distribution"],
                ["distribution"],
                ["distribution"],
                ["demand"],
                ["demand"],
                ["demand"],
                ["demand"],
                ["demand"]
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
        self.edge_upstream_nodes = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                             3, 4, 5, 6, 7, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6,
                                             7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 11,
                                             11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 14, 15, 16,
                                             17, 18
                                            ], dtype=self.int_dtype)
        self.edge_downstream_nodes = np.array([3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7,
                                               3, 4, 5, 6, 7, 8, 9, 10, 8, 9, 10, 8, 9, 10,
                                               8, 9, 10, 8, 9, 10, 11, 12, 13, 11, 12, 13,
                                               11, 12, 13, 14, 15, 16, 17, 18, 14, 15, 16, 17, 18,
                                               14, 15, 16, 17, 18, 19, 20, 21, 22, 23
                                              ],dtype=self.int_dtype)

        #define the input and output products based on the product_tags
        self.edge_input_products = np.array(
            [[1,0]]*20 + [[0,1]]*44,
            dtype=self.int_dtype,
        )

        self.edge_output_products = np.array(
            [[1,0]]*15 + [[0,1]]*49,
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
            random_factors = np.clip(self.random_param(
                loc=1.0, scale=0.1, size=self.num_timesteps
            ), 0.1, 10.0)
            self.edge_costs[i, 0, :] = -cost_edge[i] * random_factors

        # Emission costs with different random factors
        for i in range(self.num_edges):
            random_factors = np.clip(self.random_param(
                loc=1.0, scale=0.1, size=self.num_timesteps
            ), 0.1, 10.0)
            self.edge_costs[i, 1, :] = -emission_edge[i] * random_factors

        # Overwrite the last 2 edges (prices), shape = num_edge, num_cost, num_timesteps
        random_factors = np.clip(self.random_param(
            loc=1.0, scale=0.1, size=self.num_timesteps
        ), 0.1, 10.0)
        base_prices = np.array([100, 101, 105, 103, 104]) # num_cost is set only for monetary

        self.edge_costs[-self.num_market:, 0, :] = base_prices[:, np.newaxis] * random_factors[np.newaxis, :]
        
        # Start with 0 movements
        self.edge_inputs = np.zeros(
            (self.num_edges, self.num_timesteps), dtype=self.int_dtype
        )  # Start with 0 movements
        self.edge_outputs = np.zeros(
            (self.num_edges, self.num_timesteps), dtype=np.float32
        )  # Start with 0 movements
        self.node_control = np.array(
            [False]*3 + [True]*16 + [False]*5, dtype=np.bool_
        )  # Only the factory and retailer controlled by us
        self.node_inventory = np.zeros(
            (self.num_nodes, self.num_products, self.num_timesteps), dtype=self.int_dtype
        )
        
        for i in range(16):
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
            random_factors = np.clip(self.random_param(
                loc=1.0, scale=0.1, size=self.num_timesteps
            ), 0.1, 10.0)
            self.node_costs[i, :, 0, :] = -cost_node[i] * random_factors
        # Emission node costs
        for i in range(self.num_nodes):
            random_factors = np.clip(self.random_param(
                loc=1.0, scale=0.1, size=self.num_timesteps
            ), 0.1, 10.0)
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
        
        for i in range(16):
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
        base_prices = np.array([100, 101, 105, 103, 104]) # num_cost is set only for monetary

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
