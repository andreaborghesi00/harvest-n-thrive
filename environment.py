import numpy as np
import gymnasium as gym
from typing import Tuple, Any, Dict, Optional, List
from gymnasium.envs.registration import register
import math
"""
Problem Description:
In this resource management game, the agent must allocate resources such as water, fertilizer, and labour to optimize crop growth while managing limited supplies in a farm. Each turn, the environmnet dynamically changes with fluctuating weather conditions, soil health variations, and shifting market prices, requiring the agent to adapt its strategy accordingly. Overwatering may lead to crop diseases, while under-fertilization slows growth and reduces tield, depending on the crops and the weather and soil conditions. At the end of each year, the farm production is sold and this represents the reward that the agent wants to maximize in the long-term (i.e. after a certain number of years in an episodic setting).

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

WEATHER_EVENTS = {
    "sunny": {"probability": 35/52},
    "rain": {"water_amount": (0.1, 0.5), "probability":10/52 },
    "storm": {"health_reduction": (0.1, 0.3), "probability": 2/52},
    "extreme_heat": {"water_drying_factor": 0.3, "probability": 5/52},
}

LABOR_COSTS ={
    "planting": 1.0,
    "watering": 0.5,
    "fertilizing": 0.5,
}

WEATHER_DIST = [WEATHER_EVENTS[key]["probability"] for key in WEATHER_EVENTS.keys()]

class Farm(gym.Env):
    CROP_TYPES = {
        0: {"name": "wheat", "growth_time": 28, "base_yield": 0.8, "water_need": 0.6, "fertilizer_need": 0.5, "price": 10},
        1: {"name": "corn", "growth_time": 42, "base_yield": 0.9, "water_need": 0.7, "fertilizer_need": 0.6, "price": 15},
        2: {"name": "tomato", "growth_time": 35, "base_yield": 0.7, "water_need": 0.8, "fertilizer_need": 0.7, "price": 20},
        3: {"name": "potato", "growth_time": 30, "base_yield": 0.85, "water_need": 0.5, "fertilizer_need": 0.4, "price": 12},
    }
    
    def __init__(self, farm_size=(10, 10), years=10, yearly_water_supply=1000, yearly_fertilizer_supply=500, weekly_labour_supply=100):
        super().__init__()
        
        self.years = years
        self.yearly_water_supply = yearly_water_supply
        self.yearly_fertilizer_supply = yearly_fertilizer_supply
        self.weekly_labour_supply = weekly_labour_supply
        self.episode = None
        # penalties
        self.dead_crop_penalty = 3
        self.unwatered_crop_penalty = 1.5
        
        self.total_cells = farm_size[0] * farm_size[1]
        print(self.total_cells)
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
                "labour_supply": gym.spaces.Box(low=0., high=self.weekly_labour_supply, shape=(1,), dtype=np.float32),
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
        self.labour_supply = self.weekly_labour_supply
        self.current_year = 0
        self.current_week = 0
        self.inventory = np.zeros(len(self.CROP_TYPES), dtype=np.int16)
        self.info_memory = {
            "dead_crops": 0,
            "unwatered_crops": 0,
            "planted_crops": 0,
            "harvested_crops": 0,
            "water_used": 0.0,
            "fertilizer_used": 0.0,
            "growth_stage": 0.0,
            "health": 0.0,
            "yield": 0.0,
            "water_wasted": 0.0,
            "fertilizer_wasted": 0.0,
            "harvest_reward": 0.0,
            "labour_used": 0.0,
            "overwatered_crops": 0,
        }
        
    
    def _get_observation(self) -> Dict[str, Any]:
        return {
            "farm_state": self.farm,
            "water_supply": np.array([self.water_supply], dtype=np.float32),
            "fertilizer_supply": np.array([self.fertilizer_supply], dtype=np.float32),
            "labour_supply": np.array([self.labour_supply], dtype=np.float32),
            "current_week": self.current_week,
            "current_year": self.current_year,
            # "inventory": self.inventory
        }
        

    def _get_reward(self) -> float:
        pass

        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        np.random.seed(seed)
        self.episode = options.get("episode", None) if options else None
        self.farm = np.zeros(shape=(self.total_cells, 6), dtype=np.float32)  # [crop_id, growth_stage, health, yield, water, fertilizer]
        self.farm[:, 0] = -1  # No crop planted
        self.water_supply = self.yearly_water_supply
        self.fertilizer_supply = self.yearly_fertilizer_supply
        self.labour_supply = self.weekly_labour_supply
        self.current_year = 0
        self.current_week = 0
        self.labour_supply = self.weekly_labour_supply
        self.inventory = np.zeros(len(self.CROP_TYPES), dtype=np.int16)
        self.info_memory = {
            "dead_crops": 0,
            "unwatered_crops": 0,
            "planted_crops": 0,
            "harvested_crops": 0,
            "water_used": 0.0,
            "fertilizer_used": 0.0,
            "growth_stage": 0.0,
            "health": 0.0,
            "yield": 0.0,
            "water_wasted": 0.0,
            "fertilizer_wasted": 0.0,
            "harvest_reward": 0.0,
            "labour_used": 0.0,
            "overwatered_crops": 0,

        }
        return self._get_observation(), {}

    def get_info(self) -> Dict[str, Any]:
        return self.info_memory
    
    def step(self, action: Dict[str, Any]):
        crop_mask = action["crop_mask"]
        crop_selection = action["crop_selection"]
        # harvest_mask = action["harvest_mask"]
        water_amount = action["water_amount"]
        fertilizer_amount = action["fertilizer_amount"]
        reward = 0
        terminated = False
        truncated = False
        
        previous_water_supply = self.water_supply
        previous_fertilizer_supply = self.fertilizer_supply
        water_wasted = 0
        fertilizer_wasted = 0

        # Apply actions to the farm
        # Plant crops
        for i in range(self.total_cells):
            if self.labour_supply >= LABOR_COSTS["planting"]:
                if crop_mask[i] == 1 and self.farm[i, 0] == -1:
                    # Plant a new crop
                    self.farm[i, 0] = crop_selection
                    self.farm[i, 1] = 0.0 # growth stage
                    self.farm[i, 2] = 1.0 # health
                    self.farm[i, 3] = 0.0 # yield
                    
                    self.labour_supply -= LABOR_COSTS["planting"]
                    self.info_memory["planted_crops"] = self.info_memory.get("planted_crops", 0) + 1
            else: continue
        plant_costs = self.weekly_labour_supply - self.labour_supply
        prev_labour = self.labour_supply
        # print(f"Planting costs: {plant_costs}")
        # Water crops
        for i in range(self.total_cells):
            if self.labour_supply >= LABOR_COSTS["watering"]:
                if water_amount[i] > 0:
                    water_used = min(water_amount[i], self.water_supply)
                    self.farm[i, 4] += water_used
                    self.water_supply -= water_used
                    if self.farm[i, 0] == -1:
                        # if the cell is empty, water is wasted
                        water_wasted += water_used
                    
                    # clip water to 2
                    self.farm[i, 4] = np.clip(self.farm[i, 4], 0, 2)
                    
                    self.labour_supply -= LABOR_COSTS["watering"]
            else: continue
        watering_costs = prev_labour - self.labour_supply
        # print(f"Watering costs: {watering_costs}")
                    
        prev_labour = self.labour_supply
        # Fertilize crops
        for i in range(self.total_cells):     
            if self.labour_supply >= LABOR_COSTS["fertilizing"]:   
                if fertilizer_amount[i] > 0:
                    fertilizer_used = min(fertilizer_amount[i], self.fertilizer_supply)
                    self.farm[i, 5] += fertilizer_used
                    self.fertilizer_supply -= fertilizer_used
                    
                    if self.farm[i, 0] == -1:
                        # if the cell is empty, fertilizer is wasted
                        fertilizer_wasted += fertilizer_used
                    
                    # clip fertilizer to 2
                    self.farm[i, 5] = np.clip(self.farm[i, 5], 0, 2)
                    
                    self.labour_supply -= LABOR_COSTS["fertilizing"]
            else: continue

        fertilizing_costs = prev_labour - self.labour_supply
        # print(f"Fertilizing costs: {fertilizing_costs}")
    
        # Health and growth stage update
        for i in range(self.total_cells):
            if self.farm[i, 0] != -1:
                crop_id = int(self.farm[i, 0])
                growth_time = self.CROP_TYPES[crop_id]["growth_time"]
                water_need = self.CROP_TYPES[crop_id]["water_need"]
                fertilizer_need = self.CROP_TYPES[crop_id]["fertilizer_need"]
                base_yield = self.CROP_TYPES[crop_id]["base_yield"]
                growth_step = 1 / growth_time
                # print(f"Growth step: {growth_step}")
                
                if self.farm[i, 4] >= 1.8 * water_need: # TODO: handle overwatering
                    # Overwatering: growth and yield stale, health reduced
                    self.farm[i, 2] -= 0.2 # a hard hit
                    if self.farm[i, 2] <= 0: 
                        # dead crop, automatically removed
                        self.farm[i, 0] = -1 
                        self.farm[i, 1] = 0.0
                        self.farm[i, 2] = 0.0
                        self.farm[i, 3] = 0.0
                        
                        # reward -= self.dead_crop_penalty
                        
                        self.info_memory["dead_crops"] = self.info_memory.get("dead_crops", 0) + 1
                        self.info_memory["overwatered_crops"] = self.info_memory.get("overwatered_crops", 0) + 1
                    
                elif self.farm[i, 4] >= water_need: # TODO: handle overwatering
                    # Compatible with life
                    # growth, health, yield
                    growth_mutliplier = 1.5 if self.farm[i, 5] >= fertilizer_need else 1.0 # speed up growth if enough fertilizer is used
                    self.farm[i, 1] = min(self.farm[i,1] + growth_step * growth_mutliplier, 1.0)
                    self.farm[i, 2] = min(self.farm[i, 2] + 0.1, 1.0)
                    self.farm[i, 3] += base_yield * growth_step * 1.25 if self.farm[i, 5] >= fertilizer_need else base_yield * growth_step
                    # print(f"Yield: {self.farm[i, 3]}")
                else:
                    # Not enough water: growth stale, health and yield reduced
                    self.farm[i, 2] -= 0.2
                    self.farm[i, 3] -= base_yield * growth_step * .5
                    self.farm[i, 3] = max(self.farm[i, 3], 0.0)
                    # reward -= self.unwatered_crop_penalty
                    
                    self.info_memory["unwatered_crops"] = self.info_memory.get("unwatered_crops", 0) + 1
                    if self.farm[i, 2] <= 0: 
                        # dead crop, automatically removed
                        self.farm[i, 0] = -1 
                        self.farm[i, 1] = 0.0
                        self.farm[i, 2] = 0.0
                        self.farm[i, 3] = 0.0
                        
                        # reward -= self.dead_crop_penalty
                        
                        self.info_memory["dead_crops"] = self.info_memory.get("dead_crops", 0) + 1
        
        curr_water_supply = self.water_supply
        curr_fertilizer_supply = self.fertilizer_supply
        
        water_used = previous_water_supply - curr_water_supply - water_wasted
        fertilizer_used = previous_fertilizer_supply - curr_fertilizer_supply - fertilizer_wasted
        
        self.info_memory["water_used"] += (water_used)
        self.info_memory["fertilizer_used"] += (fertilizer_used)
        self.info_memory["water_wasted"] += water_wasted
        self.info_memory["fertilizer_wasted"] += fertilizer_wasted
        self.info_memory["growth_stage"] += np.mean(self.farm[:, 1]) # average growth stage
        self.info_memory["health"] += np.mean(self.farm[:, 2]) # average health
        self.info_memory["yield"] += np.sum(self.farm[:, 3]) # average yield
        self.info_memory["labour_used"] += (self.weekly_labour_supply - self.labour_supply) # total labour used
        
        # step bonus rewards
        # growth_reward = np.mean(self.farm[:, 1]) * 1.3 # reward for growing crops
        # health_reward = np.mean(self.farm[:, 2]) * 1.4 # reward for healthy crops
        yield_reward = np.sum(self.farm[:, 3]) * 1.2 # reward for yield, we really want to encourage the agent to grow crops that yield more
        
        # good management rewards
        planted_cells = np.sum(self.farm[:, 0] >= 0)
        planting_reward = planted_cells / self.total_cells * 1.4
        
        # step bonus for resource efficiency
        water_efficiency = (water_used) * 1
        fertilizer_efficiency = (fertilizer_used) * 1
        
        # penalize water waste
        water_wasted_penalty = (water_wasted) * 1
        fertilizer_wasted_penalty = (fertilizer_wasted) * 1

        # REWARD COMPUTATION
        # reward += yield_reward
        # reward += water_efficiency + fertilizer_efficiency
        # reward -= (water_wasted_penalty + fertilizer_wasted_penalty)
        # reward += planting_reward
        
        # end of week: the soil dries and the crops consume the fertilizer
        for i in range(self.total_cells):
            # get crop water and fertilizer needs
            if self.farm[i, 0] != -1:
                crop_id = int(self.farm[i, 0])
                water_need = self.CROP_TYPES[crop_id]["water_need"]
                fertilizer_need = self.CROP_TYPES[crop_id]["fertilizer_need"]
                
                # reduce water and fertilizer by the needs of the crop
                self.farm[i, 4] = max(self.farm[i, 4] - water_need, 0.0)
                self.farm[i, 5] = max(self.farm[i, 5] - fertilizer_need, 0.0)
            else:
                # empty cell, reduce water and fertilizer by 0.4
                self.farm[i, 4] = max(self.farm[i, 4] - 0.4, 0.0)
                self.farm[i, 5] = max(self.farm[i, 5] - 0.4, 0.0)
                
        
        self.current_week += 1
        if (self.current_week % 52) == 0:
            self.current_week = 0
            self.current_year += 1
            
            # Harvest crops
            for i in range(self.total_cells):
                if self.farm[i, 0] != -1:
                    harvest_reward = (self.farm[i, 3] * self.CROP_TYPES[int(self.farm[i, 0])]["price"]) * 3
                    reward += harvest_reward

                    # remove the crop from the farm
                    self.farm[i, 0] = -1
                    self.farm[i, 1] = 0.0
                    self.farm[i, 2] = 0.0
                    self.farm[i, 3] = 0.0
                    self.info_memory["harvested_crops"] += 1
                    self.info_memory["harvest_reward"] += harvest_reward
            # replenish water and fertilizer supplies... what if we don't replenish yearly but let the agent manage the supplies?
            self.water_supply = self.yearly_water_supply
            self.fertilizer_supply = self.yearly_fertilizer_supply
            
            # check if the episode is finished
            if self.current_year >= self.years:
                terminated = True
        self.labour_supply = self.weekly_labour_supply


        # next week weather event
        weather_event = np.random.choice(list(WEATHER_EVENTS.keys()), p=WEATHER_DIST)
        if weather_event == "rain":
            # Each tile receives a random amount of water (0.1 to 0.5)
            rain_amount = np.random.uniform(*WEATHER_EVENTS["rain"]["water_amount"], size=self.total_cells)
            self.farm[:, 4] += rain_amount
            self.farm[:, 4] = np.clip(self.farm[:, 4], 0, 2)
        elif weather_event == "storm":
            # Randomly damages crops, reducing their health by 0.1 to 0.3
            storm_damage = np.random.uniform(*WEATHER_EVENTS["storm"]["health_reduction"], size=self.total_cells)
            self.farm[:, 2] -= storm_damage
            self.farm[:, 2] = np.clip(self.farm[:, 2], 0, 1)
            # If health is 0, the crop is dead
            for i in range(self.total_cells):
                if self.farm[i, 2] <= 0:
                    # dead crop, automatically removed
                    self.farm[i, 0] = -1 
                    self.farm[i, 1] = 0.0
                    self.farm[i, 2] = 0.0
                    self.farm[i, 3] = 0.0
                    
                    # reward -= self.dead_crop_penalty * .5 # should i penalize this?
                    
                    self.info_memory["dead_crops"] += 1
        elif weather_event == "extreme_heat":
            # Dries soil 30% faster
            for i in range(self.total_cells):
                self.farm[i, 4] = max(self.farm[i, 4] - (0.3 * self.farm[i, 4]), 0.0)
                # reduce fertilizer by 0.1
                self.farm[i, 5] = max(self.farm[i, 5] - 0.1, 0.0)
        # else: sunny weather, no changes 

        # update observation
        observation = self._get_observation()
        observation["farm_state"] = self.farm
        observation["water_supply"] = np.array([self.water_supply], dtype=np.float32)
        observation["fertilizer_supply"] = np.array([self.fertilizer_supply], dtype=np.float32)
        observation["labour_supply"] = np.array([self.labour_supply], dtype=np.float32)
        observation["current_week"] = self.current_week
        observation["current_year"] = self.current_year
        
        return observation, reward, terminated, truncated, self.info_memory      


# register the environment
register(
    id="Farm-v0",
    entry_point=lambda  farm_size=(10, 10),
                        years=10,
                        yearly_water_supply=1000, 
                        yearly_fertilizer_supply=500, 
                        weekly_labour_supply=100: Farm(farm_size=farm_size,
                                                      years=years, 
                                                      yearly_water_supply=yearly_water_supply, 
                                                      yearly_fertilizer_supply=yearly_fertilizer_supply, 
                                                      weekly_labour_supply=weekly_labour_supply),
    max_episode_steps=2600, # 50 years 
)
        