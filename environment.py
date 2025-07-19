import numpy as np
import gymnasium as gym
from typing import Tuple, Any, Dict, Optional, List
from gymnasium.envs.registration import register
import math
from configs.config import get_config
"""
(For better readability i suggest to turn on word wrap in your editor (Alt+z in VSCode))

Implementation notes:
- The farm is represented as an array of size farm_size[0] * farm_size[1], where each element represents a crop.
- a crop is represented as a tuple (crop_id: int, growth_stage: float, health: float, yield: float/int, water: float, fertilizer: float)
    - The crop_id is an integer representing the type of crop (e.g. 0 for wheat, 1 for corn, etc.).
    - The growth_stage is a float between 0 and 1, where 0 means the crop is just planted and 1 means the crop is fully grown.
    - The health is a float between 0 and 1, where 0 means the crop is dead and 1 means the crop is healthy.
    - The yield is a float between 0 and base_yield, where 0 means no yield and base_yield means maximum yield.
    - The water is a float between 0 and 2, where 0 means no water, 1 is about right, 2 means maximum water (overflows elsewhere).
    - The fertilizer is a float between 0 and 1, where 0 means no fertilizer and 1 means maximum fertilizer.
- The agent can plant a new crop (empty tiles only), water a crop, fertilize a crop, or harvest a crop (full tiles only), or do nothing
- If the agent waters a crop too much (1.8x the water need), the crop health is reduced by 0.2, and the growth and yield are stale. (overwatering)
- If the agent doesn't meet the minimum watering criteria for a crop (underwatering), health and yield are reduced, while the growth is stale
- Once a crop is fully grown it cannot increase its yield anymore. Hence, underwatering permanently damages the potential yield of a crop

- Harvested crops are immediatly sold at the market, rewarding the agent proportionally to the yield of the crop and the current market price.
- Each crop has a specific growth time and yield, which are defined in a dictionary.
- A time step is a week of the year, and each week the crops grow, and the agent can take actions.


- Weather events: we introduce random weather events that alter the farm conditions:
    - Rain: Each tile receives a random amount of water (0.1 to 0.5)
    - Storm: Randomly damages crops, reducing their health by 0.1 to 0.3
    - Sunny: Default weather, no changes
    - Extreme Heat: Dries 30% of the water in the soil

    - The first week is always Sunny (when the environment is reset), and the weather events are drawn randomly at the end of each week, affecting the farm conditions for the next week.


- Labour: The agent has a limited amount of labour supply each week, which is used to plant, water, and fertilize crops. If the labour supply is not enough, the agent cannot perform all actions.
    - The labour gets distributed to the actions in the following order: (1) planting, (2) watering, (3) fertilizing. If there is not enough labour supply, the agent cannot perform all actions.
    - Labour costs: 
        - planting a crop costs 1 labour
        - watering a crop costs 0.5 labour
        - fertilizing a crop costs 0.5 labour. 
        - harvesting a crop costs 0.75 labour.
    ideally we won't allow the agent to have more than 2.75 * farm_size[0] * farm_size[1] labour supply, so that the agent can perform all actions in a week.

- Oscillating prices: the market prices for crops oscillate over time, with a period of 52 weeks (1 year). For simplicity's sake the prices of each type of crop will oscillate like a cosine function and each of these functions will have their phase shifted by a uniform amount. E.g. with 4 crops we will have a favorite crop per season. 

"""
CONFIGS = get_config()

WEATHER_EVENTS = {
    "sunny": {"probability": 35./52.},
    "rain": {"water_amount": (0.1, 0.5), "probability":10./52. },
    "storm": {"health_reduction": (0.1, 0.3), "probability": 2./52.},
    "extreme_heat": {"water_drying_factor": 0.3, "probability": 5./52.},
}

LABOR_COSTS ={
    "planting": CONFIGS.planting_lab_cost,
    "watering": CONFIGS.watering_lab_cost,
    "fertilizing": CONFIGS.fertilizing_lab_cost,
    "harvesting": CONFIGS.harvesting_lab_cost,
}

WEATHER_DIST = [WEATHER_EVENTS[key]["probability"] for key in WEATHER_EVENTS.keys()]

"""
name: name of the crop
growth_time: time in weeks for the crop to grow from 0 to base_yield
base_yield: the yield of a fully grown crop with no fertilizer
water_need: the amount of water needed each week for the crop to grow
fertilizer_need: the amount of fertilizer needed each week for the crop to grow 50% faster that week and yield 50% more
price: the base price of the crop at the market with yield 1.0 
"""
CROP_TYPES = {
    1: {"name": "wheat", "growth_time": 8, "base_yield": 0.8, "water_need": 0.6, "fertilizer_need": 0.5, "price": 10},
    2: {"name": "corn", "growth_time": 4, "base_yield": 0.9, "water_need": 0.7, "fertilizer_need": 0.6, "price": 15},
    3: {"name": "tomato", "growth_time": 5, "base_yield": 0.7, "water_need": 0.8, "fertilizer_need": 0.7, "price": 20},
    4: {"name": "potato", "growth_time": 6, "base_yield": 0.85, "water_need": 0.5, "fertilizer_need": 0.4, "price": 12},
}

class Farm(gym.Env):
    
    def __init__(self, farm_size=(10, 10),
                 years=10, 
                 yearly_water_supply=1000, 
                 yearly_fertilizer_supply=500, 
                 weekly_labour_supply=100,
                 exp=2.0):
        super().__init__()
        self.exp = exp 
        self.years = years
        self.yearly_water_supply = yearly_water_supply
        self.yearly_fertilizer_supply = yearly_fertilizer_supply
        self.weekly_labour_supply = weekly_labour_supply
        self.total_cells = farm_size[0] * farm_size[1]
        self.market_high = 1.5 # maximum market price multiplier
        self.market_low = 0.5 # minimum market price multiplier
        self.unwatered_crop_penalty = 1.5  # penalty for unwatered crops
        self.dead_crop_penalty = 3.0 # penalty for dead crops
        self.action_space = gym.spaces.Dict(
            {
                # "crop_mask": gym.spaces.MultiDiscrete([len(CROP_TYPES) + 1] * self.total_cells),  # +1 for no crop selected
                "crop_mask": gym.spaces.MultiBinary(self.total_cells),  #
                "crop_type": gym.spaces.Discrete(len(CROP_TYPES), start=1),  # 1 to len(CROP_TYPES) inclusive
                "harvest_mask": gym.spaces.MultiBinary(self.total_cells),
                "water_amount": gym.spaces.Box(low=0., high=1., shape=(self.total_cells,), dtype=np.float32),
                "fertilizer_amount": gym.spaces.Box(low=0., high=1., shape=(self.total_cells,), dtype=np.float32),
            }
        )
        
        self.observation_space = gym.spaces.Dict(
            {
                # [crop_id, growth_stage, health, yield, water, fertilizer]
                "farm_state": gym.spaces.Box(low=np.array([0., 0., 0., 0., 0., 0.] * self.total_cells, dtype=np.float32).reshape(self.total_cells, 6),
                                             high=np.array([len(CROP_TYPES), 1., 1., 1., 2., 1.] * self.total_cells, dtype=np.float32).reshape(self.total_cells, 6),
                                             shape=(self.total_cells, 6),
                                             dtype=np.float32),
                "water_supply": gym.spaces.Box(low=0., high=self.yearly_water_supply, shape=(1,), dtype=np.float32),
                "fertilizer_supply": gym.spaces.Box(low=0., high=self.yearly_fertilizer_supply, shape=(1,), dtype=np.float32),
                "labour_supply": gym.spaces.Box(low=0., high=self.weekly_labour_supply, shape=(1,), dtype=np.float32),
                "current_week": gym.spaces.Discrete(52),
                "current_year": gym.spaces.Discrete(self.years),
                "market_multipliers": gym.spaces.Box(low=self.market_low, high=self.market_high, shape=(len(CROP_TYPES),), dtype=np.float32),
            }
        )
        
        self.farm = np.zeros(shape=(self.total_cells, 6), dtype=np.float32)  # [crop_id, growth_stage, health, yield, water, fertilizer]
        self.water_supply = self.yearly_water_supply
        self.fertilizer_supply = self.yearly_fertilizer_supply
        self.labour_supply = self.weekly_labour_supply
        self.current_year = 0
        self.current_week = 0
        self.harvest_count = 0
        # self.inventory = np.zeros(len(CROP_TYPES), dtype=np.int16)
        
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
        
    def _get_market_multipliers(self) -> List[float]:
        """
        Calculate market multipliers for each crop type based on the current week.
        It's simply a cosine function that oscillates between market_high and market_low with a phase shift for each crop type. The distance between each phase is uniformly distributed across the number of crop types.
        """
        return np.array([
            ((self.market_high + self.market_low)/2) + 
            ((self.market_high - self.market_low)/2) * np.cos((self.current_week / 52) * (2 * np.pi) + (crop_id * ((2*np.pi) / len(CROP_TYPES))))
            for crop_id in range(len(CROP_TYPES))
        ], dtype=np.float32)
    
    def _get_observation(self) -> Dict[str, Any]:
        return {
            "farm_state": self.farm,
            "water_supply": np.array([self.water_supply], dtype=np.float32),
            "fertilizer_supply": np.array([self.fertilizer_supply], dtype=np.float32),
            "labour_supply": np.array([self.labour_supply], dtype=np.float32),
            "current_week": self.current_week,
            "current_year": self.current_year,
            "market_multipliers": self._get_market_multipliers(),
            # "inventory": self.inventory
        }
        

    def _get_reward(self) -> float:
        pass

        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        np.random.seed(seed)
        self.farm = np.zeros(shape=(self.total_cells, 6), dtype=np.float32)  # [crop_id, growth_stage, health, yield, water, fertilizer]
        self.farm[:, 2] = 1.0  # Set initial health to 1.0 for all crops
        self.water_supply = self.yearly_water_supply
        self.fertilizer_supply = self.yearly_fertilizer_supply
        self.labour_supply = self.weekly_labour_supply
        self.current_year = 0
        self.current_week = 0
        self.labour_supply = self.weekly_labour_supply
        # self.inventory = np.zeros(len(CROP_TYPES), dtype=np.int16)
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
        self.harvest_count = 0
        return self._get_observation(), {}

    def get_info(self) -> Dict[str, Any]:
        return self.info_memory
    
    def step(self, action: Dict[str, Any]):
        crop_mask = action["crop_mask"]
        harvest_mask = action["harvest_mask"]
        water_amount = action["water_amount"]
        fertilizer_amount = action["fertilizer_amount"]
        crop_selection = action["crop_type"]

        reward = 0
        terminated = False
        truncated = False
        
        previous_water_supply = self.water_supply
        previous_fertilizer_supply = self.fertilizer_supply
        water_wasted = 0
        fertilizer_wasted = 0

        ### Apply actions to the farm ###
        # Plant crops
        plant_mask = (crop_mask > 0) & (self.farm[:, 0] == 0)  # Only plant in empty cells and where crop_mask is greater than 0
        max_ops = int(self.labour_supply // LABOR_COSTS["planting"]) # maximum number of operations we can perform given current labour supply
        plant_indices = np.nonzero(plant_mask)[0]  # Get indices of cells where we can plant crops
        to_plant = plant_indices[:max_ops]  # Limit to the maximum number of operations we can perform
        
        self.farm[to_plant, 0] = crop_selection            # crop_id
        self.farm[to_plant, 1] = 0.0                             # growth_stage
        self.farm[to_plant, 2] = 1.0                             # health
        self.farm[to_plant, 3] = 0.0                             # yield

        num_planted = to_plant.shape[0]
        self.labour_supply -= num_planted * LABOR_COSTS["planting"]
        self.info_memory["planted_crops"] = self.info_memory.get("planted_crops", 0) + num_planted        
        
        # Watering crops
        want_water_mask = water_amount > 0               
        candidate_idxs  = np.nonzero(want_water_mask)[0]  

        max_ops = int(self.labour_supply // LABOR_COSTS["watering"]) 
        to_water = candidate_idxs[:max_ops] # at most max_ops cells

        req = water_amount[to_water] # gather requested amounts shape (total_cells,)

        #  compute actual water_used sequentially via vectorized cumsum trick 
        #  for each j: water_used[j] = min(req[j], remaining_supply_before_j)
        cumsum_req = np.cumsum(req) # [r0, r0+r1, …] shape is still (totla_cells,)
        remaining_before = self.water_supply - (cumsum_req - req) # we remove req as this is the amount before watering when watering cell j 
        water_used = np.minimum(req, np.clip(remaining_before, 0, None))  # cumsum trick

        self.farm[to_water, 4] += water_used
        self.farm[:, 4] = np.clip(self.farm[:, 4], 0, 2)

        self.water_supply -=  water_used.sum()

        empty_mask = (self.farm[to_water, 0] == 0) # empty watered cells
        water_wasted += water_used[empty_mask].sum()

        num_watered = to_water.shape[0]
        self.labour_supply -= num_watered * LABOR_COSTS["watering"]

        # Fertilizing crops                    
        want_fertilizer_mask = fertilizer_amount > 0               
        candidate_idxs  = np.nonzero(want_fertilizer_mask)[0]  

        max_ops = int(self.labour_supply // LABOR_COSTS["fertilizing"]) 
        to_fertilize = candidate_idxs[:max_ops] # at most max_ops cells

        req = fertilizer_amount[to_fertilize] # gather requested amounts shape (total_cells,)

        #  compute actual fertilizer_used sequentially via vectorized cumsum trick 
        #  for each j: fertilizer_used[j] = min(req[j], remaining_supply_before_j)
        cumsum_req = np.cumsum(req) # [r0, r0+r1, …] shape is still (totla_cells,)
        remaining_before = self.fertilizer_supply - (cumsum_req - req) # we remove req as this is the amount before fertilizing when fertilizing cell j 
        fertilizer_used = np.minimum(req, np.clip(remaining_before, 0, None))  # cumsum trick

        self.farm[to_fertilize, 5] += fertilizer_used
        self.farm[:, 5] = np.clip(self.farm[:, 5], 0, 1)

        self.fertilizer_supply -=  fertilizer_used.sum()

        empty_mask = (self.farm[to_fertilize, 0] == 0) # empty fertilized cells
        fertilizer_wasted += fertilizer_used[empty_mask].sum()

        num_fertilized = to_fertilize.shape[0]
        self.labour_supply -= num_fertilized * LABOR_COSTS["fertilizing"] 
        

        # Health and growth stage update
        for i in range(self.total_cells):
            if self.farm[i, 0] > 0:
                crop_id = int(self.farm[i, 0])
                growth_time = CROP_TYPES[crop_id]["growth_time"]
                water_need = CROP_TYPES[crop_id]["water_need"]
                fertilizer_need = CROP_TYPES[crop_id]["fertilizer_need"]
                max_yield = CROP_TYPES[crop_id]["base_yield"] ** (1 / self.exp)  # base yield adjusted by the exponent
                growth_step = 1 / growth_time
                # print(f"Growth step: {growth_step}")
                
                if self.farm[i, 4] >= 1.8 * water_need: # TODO: handle overwatering
                    # Overwatering: growth and yield stale, health reduced
                    self.farm[i, 2] -= 0.2 # a hard hit
                    if self.farm[i, 2] <= 0: 
                        # dead crop, automatically removed
                        self.farm[i, 0] = 0 
                        self.farm[i, 1] = 0.0
                        self.farm[i, 2] = 0.0
                        self.farm[i, 3] = 0.0
                        
                        reward -= self.dead_crop_penalty
                        
                        self.info_memory["dead_crops"] = self.info_memory.get("dead_crops", 0) + 1
                        self.info_memory["overwatered_crops"] = self.info_memory.get("overwatered_crops", 0) + 1
                    
                elif self.farm[i, 4] >= water_need:
                    # Compatible with life
                    # growth, health, yield
                    growth_mutliplier = 1.5 if self.farm[i, 5] >= fertilizer_need else 1.0 # speed up growth if enough fertilizer is used
                    self.farm[i, 1] = min(self.farm[i,1] + growth_step * growth_mutliplier, 1.0)
                    self.farm[i, 2] = min(self.farm[i, 2] + 0.1, 1.0)
                    if self.farm[i, 1] <= 1.0: # the yield increses only if the crop is not fully grown 
                        self.farm[i, 3] = min(self.farm[i,3] + max_yield * growth_step * 1.5 if self.farm[i, 5] >= fertilizer_need else max_yield * growth_step, max_yield)
                else:
                    # Not enough water: growth stale, health and yield reduced
                    # This is a way to permanently damage the crop, as the yield is reduced while the growth is stale, meaning the crop will never reach its full yield potential 
                    self.farm[i, 2] = max(self.farm[i, 2] - 0.2, 0.0)
                    self.farm[i, 3] = max(self.farm[i, 3] - max_yield * growth_step * .5, 0.0)
                    reward -= self.unwatered_crop_penalty
                    
                    self.info_memory["unwatered_crops"] = self.info_memory.get("unwatered_crops", 0) + 1
                    if self.farm[i, 2] <= 0: 
                        # dead crop, automatically removed
                        self.farm[i, 0] = 0 
                        self.farm[i, 1] = 0.0
                        self.farm[i, 2] = 1.0
                        self.farm[i, 3] = 0.0
                        
                        reward -= self.dead_crop_penalty # double penalty here, death + unwatered
                        
                        self.info_memory["dead_crops"] = self.info_memory.get("dead_crops", 0) + 1
        
        curr_water_supply = self.water_supply
        curr_fertilizer_supply = self.fertilizer_supply
        
        water_used = previous_water_supply - curr_water_supply - water_wasted
        fertilizer_used = previous_fertilizer_supply - curr_fertilizer_supply - fertilizer_wasted
        
        self.info_memory["water_used"] += (water_used)
        self.info_memory["fertilizer_used"] += (fertilizer_used)
        self.info_memory["water_wasted"] += water_wasted
        self.info_memory["fertilizer_wasted"] += fertilizer_wasted
        self.info_memory["labour_used"] += (self.weekly_labour_supply - self.labour_supply) # total labour used
        

        # step bonus for resource efficiency
        water_efficiency = (water_used) * 1
        fertilizer_efficiency = (fertilizer_used) * 1
        
        # penalize water waste
        water_wasted_penalty = (water_wasted) * 1
        fertilizer_wasted_penalty = (fertilizer_wasted) * 1

        # unused labour supply
        unused_labour_penalty = (self.weekly_labour_supply - self.labour_supply) * 0.5
        
        # used labour bonus
        labour_efficiency = (self.weekly_labour_supply - self.labour_supply) * 0.5
        
        # STEP-WISE REWARD COMPUTATION
        reward += water_efficiency + fertilizer_efficiency + labour_efficiency
        reward -= (water_wasted_penalty + fertilizer_wasted_penalty)
        reward -= unused_labour_penalty
        
        # end of week: the soil dries and the crops consume the fertilizer
        for i in range(self.total_cells):
            # get crop water and fertilizer needs
            if self.farm[i, 0] > 0:
                crop_id = int(self.farm[i, 0]) 
                water_need = CROP_TYPES[crop_id]["water_need"]
                fertilizer_need = CROP_TYPES[crop_id]["fertilizer_need"]
                
                # reduce water and fertilizer by the needs of the crop
                self.farm[i, 4] = max(self.farm[i, 4] - water_need, 0.0)
                self.farm[i, 5] = max(self.farm[i, 5] - fertilizer_need, 0.0)
            else:
                # empty cell, reduce water and fertilizer
                self.farm[i, 4] = max(self.farm[i, 4] - 0.4, 0.0)
                self.farm[i, 5] = max(self.farm[i, 5] - 0.4, 0.0)
                
        
        # Harvest crops
        mm = self._get_market_multipliers()
        for i in range(self.total_cells):
            if self.labour_supply >= LABOR_COSTS["harvesting"]:
                if harvest_mask[i] == 1 and self.farm[i, 0] > 0:
                    growth_reward = np.sum(self.farm[i, 1]) * 1 # reward for growing crops
                    health_reward = np.sum(self.farm[i, 2]) * 1 # reward for healthy crops
                    yield_reward = np.sum(self.farm[i, 3]) * 10 # reward for yield, we really want to encourage the agent to grow crops that yield more
                    
                    crop_id = int(self.farm[i, 0])  # 0 is no crop, so we subtract 1 to match the CROP_TYPES index
                    harvest_reward = (self.farm[i, 3]**self.exp * CROP_TYPES[crop_id]["price"]) * mm[crop_id-1] * 3  # yield * price * market multiplier
                    # harvest_reward = (self.farm[i, 3] * CROP_TYPES[crop_id]["price"])  * 3 # yield * price 
                    reward += harvest_reward 
                    reward += yield_reward ** self.exp + health_reward + growth_reward

                    # update info memory running averages
                    self.harvest_count += 1
                    self.info_memory["growth_stage"] += (self.farm[i, 1]- self.info_memory["growth_stage"]) / self.harvest_count
                    self.info_memory["health"] += (self.farm[i, 2] - self.info_memory["health"]) / self.harvest_count
                    self.info_memory["yield"] += (self.farm[i, 3] - self.info_memory["yield"]) / self.harvest_count
                    
                    # remove the crop from the farm
                    self.farm[i, 0] = 0
                    self.farm[i, 1] = 0.0
                    self.farm[i, 2] = 1.0
                    self.farm[i, 3] = 0.0
                    
                    self.labour_supply -= LABOR_COSTS["harvesting"]
                    
                    self.info_memory["harvested_crops"] += 1
                    self.info_memory["harvest_reward"] += harvest_reward
            else: break
                
        self.current_week += 1
        self.current_week %= 52
        if (self.current_week % 52) == 0:
            # replenish water and fertilizer supplies... what if we don't replenish yearly but let the agent manage the supplies?
            self.current_year += 1
            
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
            planted_mask = self.farm[:, 0] > 0  # Only apply storm damage to planted crops
            self.farm[planted_mask, 2] -= storm_damage[planted_mask]
            # If health is 0, the crop is dead
            for i in range(self.total_cells):
                if self.farm[i, 0] > 0 and self.farm[i, 2] <= 0:
                    # dead crop, automatically removed
                    self.farm[i, 0] = 0 
                    self.farm[i, 1] = 0.0
                    self.farm[i, 2] = 1.0
                    self.farm[i, 3] = 0.0
                    
                    # reward -= self.dead_crop_penalty * .5 # should i penalize this? i mean it just got unlucky
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
        observation["market_multipliers"] = mm
        
        return observation, reward, terminated, truncated, self.info_memory      


# register the environment
register(
    id="Farm-v0",
    entry_point=lambda  farm_size=(10, 10),
                        years=10,
                        yearly_water_supply=1000, 
                        yearly_fertilizer_supply=500, 
                        weekly_labour_supply=100,
                        exp=2.0
                        : Farm(farm_size=farm_size, 
                               years=years, 
                               yearly_water_supply=yearly_water_supply, 
                               yearly_fertilizer_supply=yearly_fertilizer_supply, 
                               weekly_labour_supply=weekly_labour_supply,
                               exp=exp),
    max_episode_steps=2600, # 50 years 
)
        