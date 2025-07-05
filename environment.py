import numpy as np
import gymnasium as gym
from typing import Tuple, Any, Dict, Optional, List
from gymnasium.envs.registration import register

"""
Problem Description:
In this resource management game, the agent must allocate resources such as water, fertilizer, and labor to optimize crop growth while managing limited supplies in a farm. Each turn, the environmnet dynamically changes with fluctuating weather conditions, soil health variations, and shifting market prices, requiring the agent to adapt its strategy accordingly. Overwatering may lead to crop diseases, while under-fertilization slows growth and reduces tield, depending on the crops and the weather and soil conditions. At the end of each year, the farm production is sold and this represents the reward that the agent wants to maximize in the long-term (i.e. after a certain number of years in an episodic setting).

Implementation notes:
- The farm is represented as an array of size farm_size[0] * farm_size[1], where each element represents a crop.
- a crop is represented as a tuple (crop_id: int, growth_stage: float, health: float, yield: float/int, water: float, fertilizer: float)
    - The crop_id is an integer representing the type of crop (e.g. 0 for wheat, 1 for corn, etc.).
    - The growth_stage is a float between 0 and 1, where 0 means the crop is just planted and 1 means the crop is fully grown.
    - The health is a float between 0 and 1, where 0 means the crop is dead and 1 means the crop is healthy.
    - The yield is a float between 0 and 1, where 0 means no yield and 1 means maximum yield.
    - The water is a float between 0 and 1, where 0 means no water and 1 means maximum water.
    - The fertilizer is a float between 0 and 1, where 0 means no fertilizer and 1 means maximum fertilizer.
- The agent can water a crop, fertilize a crop, or harvest a crop, or do nothing
- The agent can also choose to plant a new crop in an empty field.
- The harvested crops are stored in a separate array, and at the end of each year the stored crops are sold, rewarding the agent.
- Each crop has a specific growth time and yield, which are defined in a dictionary.
- The time passes in weeks, and each week the crops grow, and the agent can take actions.
"""
class Farm(gym.Env):
    CROP_TYPES = {
        0: {"name": "wheat", "growth_time": 28, "base_yield": 0.8, "water_need": 0.6, "fertilizer_need": 0.5, "price": 10},
        1: {"name": "corn", "growth_time": 42, "base_yield": 0.9, "water_need": 0.7, "fertilizer_need": 0.6, "price": 15},
        2: {"name": "tomato", "growth_time": 35, "base_yield": 0.7, "water_need": 0.8, "fertilizer_need": 0.7, "price": 20},
        3: {"name": "potato", "growth_time": 30, "base_yield": 0.85, "water_need": 0.5, "fertilizer_need": 0.4, "price": 12},
    }
    
    def __init__(self, farm_size=(10, 10), years=10, yearly_water_supply=1000, yearly_fertilizer_supply=500, yearly_labor_supply=100):
        super().__init__()
        
        self.years = years
        self.yearly_water_supply = yearly_water_supply
        self.yearly_fertilizer_supply = yearly_fertilizer_supply
        self.yearly_labor_supply = yearly_labor_supply
        
        # penalties
        self.dead_crop_penalty = 1
        self.unwatered_crop_penalty = 0.5
        
        self.total_cells = farm_size[0] * farm_size[1]
        self.action_space = gym.spaces.Dict(
            {
                "crop_mask": gym.spaces.MultiBinary(self.total_cells),
                "crop_selection": gym.spaces.Discrete(len(self.CROP_TYPES), start=-1),
                # "harvest_mask": gym.spaces.MultiBinary(self.total_cells),
                "water_amount": gym.spaces.Box(low=0., high=1., shape=(self.total_cells,), dtype=np.float32),
                "fertilizer_amount": gym.spaces.Box(low=0., high=1., shape=(self.total_cells,), dtype=np.float32),
            }

        )
        self.observation_space = gym.spaces.Dict(
            {
                # [crop_id, growth_stage, health, yield, water, fertilizer]
                "farm_state": gym.spaces.Box(low=np.array([-1, 0., 0., 0., 0., 0.] * self.total_cells, dtype=np.float32).reshape(self.total_cells, 6),
                                             high=np.array([len(self.CROP_TYPES), 1., 1., 1., 1., 1.] * self.total_cells, dtype=np.float32).reshape(self.total_cells, 6),
                                             shape=(self.total_cells, 6),
                                             dtype=np.float32),
                "water_supply": gym.spaces.Box(low=0., high=self.yearly_water_supply, shape=(1,), dtype=np.float32),
                "fertilizer_supply": gym.spaces.Box(low=0., high=self.yearly_fertilizer_supply, shape=(1,), dtype=np.float32),
                "labor_supply": gym.spaces.Box(low=0., high=self.yearly_labor_supply, shape=(1,), dtype=np.float32),
                "current_week": gym.spaces.Discrete(52),
                "current_year": gym.spaces.Discrete(self.years),
                # "inventory": gym.spaces.Box(low=np.zeros(len(self.CROP_TYPES)),
                #                             high=np.ones(len(self.CROP_TYPES)) * 1000,
                #                             shape=(len(self.CROP_TYPES),),
                #                             dtype=np.int16)
            }
        )
        
        self.farm = np.zeros(shape=(self.total_cells, 6), dtype=np.float32)  # [crop_id, growth_stage, health, yield, water, fertilizer]
        self.water_supply = self.yearly_water_supply
        self.fertilizer_supply = self.yearly_fertilizer_supply
        self.labor_supply = self.yearly_labor_supply
        self.current_year = 0
        self.current_week = 0
        self.inventory = np.zeros(len(self.CROP_TYPES), dtype=np.int16)
        
    
    def _get_observation(self) -> Dict[str, Any]:
        return {
            "farm_state": self.farm,
            "water_supply": np.array([self.water_supply], dtype=np.float32),
            "fertilizer_supply": np.array([self.fertilizer_supply], dtype=np.float32),
            "labor_supply": np.array([self.labor_supply], dtype=np.float32),
            "current_week": self.current_week,
            "current_year": self.current_year,
            # "inventory": self.inventory
        }
        

    def _get_reward(self) -> float:
        pass

        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        np.random.seed(seed)
        
        self.farm = np.zeros(shape=(self.total_cells, 6), dtype=np.float32)  # [crop_id, growth_stage, health, yield, water, fertilizer]
        self.farm[:, 0] = -1  # No crop planted
        self.water_supply = self.yearly_water_supply
        self.fertilizer_supply = self.yearly_fertilizer_supply
        self.labor_supply = self.yearly_labor_supply
        self.current_year = 0
        self.current_week = 0
        self.inventory = np.zeros(len(self.CROP_TYPES), dtype=np.int16)
        return self._get_observation(), {}

    
    def step(self, action: Dict[str, Any]):
        crop_mask = action["crop_mask"]
        crop_selection = action["crop_selection"]
        # harvest_mask = action["harvest_mask"]
        water_amount = action["water_amount"]
        fertilizer_amount = action["fertilizer_amount"]
        reward = 0
        terminated = False
        truncated = False
        
        # Weather and soil conditions
        self.farm[:, 4] = np.clip(self.farm[:, 4] - 0.15, 0, 1)  # water evaporation
        self.farm[:, 5] = np.clip(self.farm[:, 5] - 0.2 , 0, 1)  # fertilizer degradation        
        
        # Apply actions to the farm
        for i in range(self.total_cells):
            if crop_mask[i] == 1 and self.farm[i, 0] == -1:
                # Plant a new crop
                self.farm[i, 0] = crop_selection
                self.farm[i, 1] = 0.0 # growth stage
                self.farm[i, 2] = 1.0 # health
                self.farm[i, 3] = 0.0 # yield
            # add water and fertilizer
            if water_amount[i] > 0:
                water_used = min(water_amount[i], self.water_supply)
                self.farm[i, 4] += water_used
                self.water_supply -= water_used
                
            if fertilizer_amount[i] > 0:
                fertilizer_used = min(fertilizer_amount[i], self.fertilizer_supply)
                self.farm[i, 5] += fertilizer_used
                self.fertilizer_supply -= fertilizer_used

            # clip the water and fertilizer to 1
            self.farm[i, 4] = np.clip(self.farm[i, 4], 0, 1)
            self.farm[i, 5] = np.clip(self.farm[i, 5], 0, 1)
    
        # Health and growth stage update
        for i in range(self.total_cells):
            if self.farm[i, 0] != -1:
                crop_id = int(self.farm[i, 0])
                growth_time = self.CROP_TYPES[crop_id]["growth_time"]
                water_need = self.CROP_TYPES[crop_id]["water_need"]
                fertilizer_need = self.CROP_TYPES[crop_id]["fertilizer_need"]
                base_yield = self.CROP_TYPES[crop_id]["base_yield"]
                growth_step = 1 / growth_time
                
                if self.farm[i, 4] >= water_need:
                    # Compatible with life
                    # growth, health, yield
                    growth_mutliplier = 1.5 if self.farm[i, 5] >= fertilizer_need else 1.0 # speed up growth if enough fertilizer is used
                    self.farm[i, 1] = min(self.farm[i,1] + growth_step * growth_mutliplier, 1.0)
                    self.farm[i, 2] = min(self.farm[i, 2] + 0.1, 1.0)
                    self.farm[i, 3] += base_yield * growth_step * 1.25 if self.farm[i, 5] >= fertilizer_need else base_yield * growth_step
                else:
                    # Not enough water: growth stale, health and yield reduced
                    self.farm[i, 2] -= 0.2
                    self.farm[i, 3] -= base_yield * growth_step * .5
                    self.farm[i, 3] = max(self.farm[i, 3], 0.0)
                    reward -= self.unwatered_crop_penalty
                    if self.farm[i, 2] <= 0: 
                        # dead crop, automatically removed
                        self.farm[i, 0] = -1 
                        self.farm[i, 1] = 0.0
                        self.farm[i, 2] = 0.0
                        self.farm[i, 3] = 0.0
                        
                        reward -= self.dead_crop_penalty
        
        # step bonus rewards
        growth_reward = np.mean(self.farm[:, 1]) * 1.3 # reward for growing crops
        health_reward = np.mean(self.farm[:, 2]) * 1.4 # reward for healthy crops
        yield_reward = np.mean(self.farm[:, 3]) * 1.2 # reward for yield
        
        # good management rewards
        planted_cells = np.sum(self.farm[:, 0] >= 0)
        planting_reward = planted_cells / self.total_cells * 1.2
        
        # step bonus for resource efficiency
        water_efficiency = (self.water_supply / self.yearly_water_supply) * 1.1 # reward for efficient water use
        fertilizer_efficiency = (self.fertilizer_supply / self.yearly_fertilizer_supply) * 1.1
        
        reward += health_reward + growth_reward + yield_reward
        reward += water_efficiency + fertilizer_efficiency
        reward += planting_reward
        
        # Time step # TODO: a "partial" reward comes after each week when with the harvest mask, a non-sparse reward is better for monte carlo methods
        self.current_week += 1
        if (self.current_week % 52) == 0:
            self.current_week = 0
            self.current_year += 1
            
            # Harvest crops
            for i in range(self.total_cells):
                if self.farm[i, 0] != -1:
                    reward += self.farm[i, 3] * self.CROP_TYPES[int(self.farm[i, 0])]["price"]

                    # remove the crop from the farm
                    self.farm[i, 0] = -1
                    self.farm[i, 1] = 0.0
                    self.farm[i, 2] = 0.0
                    self.farm[i, 3] = 0.0
                
            self.water_supply = self.yearly_water_supply
            self.fertilizer_supply = self.yearly_fertilizer_supply
            self.labor_supply = self.yearly_labor_supply
            
            # check if the episode is finished
            if self.current_year >= self.years:
                terminated = True


        # update observation
        observation = self._get_observation()
        observation["farm_state"] = self.farm
        observation["water_supply"] = np.array([self.water_supply], dtype=np.float32)
        observation["fertilizer_supply"] = np.array([self.fertilizer_supply], dtype=np.float32)
        observation["labor_supply"] = np.array([self.labor_supply], dtype=np.float32)
        observation["current_week"] = self.current_week
        observation["current_year"] = self.current_year
        
        return observation, reward, terminated, truncated, {}        
        
        

# register the environment
register(
    id="Farm-v0",
    entry_point=lambda  farm_size=(10, 10),
                        years=10,
                        yearly_water_supply=1000, 
                        yearly_fertilizer_supply=500, 
                        yearly_labor_supply=100: Farm(farm_size=farm_size,
                                                      years=years, 
                                                      yearly_water_supply=yearly_water_supply, 
                                                      yearly_fertilizer_supply=yearly_fertilizer_supply, 
                                                      yearly_labor_supply=yearly_labor_supply),
    max_episode_steps=2600, # 50 years 
)
        