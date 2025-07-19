from dataclasses import dataclass, field
from typing import List, Dict, Any
import yaml
import os
from pathlib import Path
import logging
import random

@dataclass(frozen=True)
class ModelConfig:
    experiment_name: str = field(default=None)
    notes: str = field(default=None)
    device: str = field(default="cuda:0")
    
    # Training hyperparameters
    batch_size: int = 10
    episodes: int = 1000
    learning_rate: float = 3e-4
    patience: int = 0
    gamma: float = 0.99
    
    ## Farm hyperparameters
    years: int = 10
    farm_size: int = 10
    weekly_water_tile_supply: float = 0.75
    weekly_fertilizer_tile_supply: float = 0.5
    weekly_labour_tile_supply: float = 1.5
    yield_exponent: float = 2.0 
    planting_lab_cost: float = 1.0
    watering_lab_cost: float = 0.5
    fertilizing_lab_cost: float = 0.5
    harvesting_lab_cost: float = 0.75
    
    ## Exploration strategies
    eps_greedy: bool = False
    icm: bool = False
    entropy_bonus: bool = False
    
    ### Exploration strategies hyperparameters
    icm_eta: float = 100.0
    
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 0.997 # to reach 0.05 needs ~1000 episodes. 0.998 for ~1500 episodes
    
    beta_start: float = 1.0
    beta_end: float = 0.05
    beta_decay: float= 0.997
    
    
    # Paths and Logging
    results_path: str = Path("results/")
    rewards_path: str =results_path / "rewards/"
    infos_path: str =results_path / "infos/"
    configs_path: str =results_path / "configs/"
        
    @classmethod
    def load_config(cls, config_path: str = os.path.join(Path(__file__).parent, "config.yaml")) -> "ModelConfig":
        # log current working directory
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at path {config_path}")
        
        with open(config_path, "r") as file:
            config_dict = yaml.safe_load(file)
        return cls(**config_dict)

# singleton
_config_instance = None

def get_config(config_path: str = None):
    global _config_instance
    if _config_instance is None:
        if config_path is None:
            config_path = os.path.join(Path(__file__).parent, "config.yaml") # default config path
        _config_instance = ModelConfig.load_config(config_path)
    
    return _config_instance

def reset_config():
    global _config_instance
    _config_instance = None
        