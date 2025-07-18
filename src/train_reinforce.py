import environment as env
from agents.reinforce import PolicyGradientAgent
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import torch
import signal
import threading
from pathlib import Path
from configs.config import get_config

CONFIGS = get_config()

Path(CONFIGS.rewards_path).mkdir(parents=True, exist_ok=True)
Path(CONFIGS.infos_path).mkdir(parents=True, exist_ok=True)
Path(CONFIGS.configs_path).mkdir(parents=True, exist_ok=True)

shutdown_flag = threading.Event()
sigusr1_flag = threading.Event()

def on_sigint(signum, frame):
    print("\nSIGINT received; will stop after this episode.")
    shutdown_flag.set()

def on_sigusr1(signum, frame):
    print("\nSIGUSR1 received; will save current state.")
    sigusr1_flag.set()
    
signal.signal(signal.SIGINT, on_sigint)
signal.signal(signal.SIGUSR1, on_sigusr1)


def flatten_observation(obs):
    return np.concatenate([
        obs['farm_state'].flatten(),
        obs['water_supply'],
        obs['fertilizer_supply'],
        obs['labour_supply'],
        [obs['current_week'] / 52],
        [obs['current_year'] / CONFIGS.years]
    ])
    
def flatten_action(action):
    return np.concatenate([
        action['crop_mask'].flatten(),
        [action['crop_type']],
        action['harvest_mask'].flatten(),
        action['water_amount'].flatten(),
        action['fertilizer_amount'].flatten()
    ])


def average_infos(infos):
    avg_info = {}
    for key in infos[0].keys():
        avg_info[key] = np.mean([info[key] for info in infos])
    return avg_info


def random_train(farm_env, episodes):
    pbar = tqdm(total=episodes, desc="Training", unit="step")
    mean_rewards = []
    farm_env.reset()
    infos = []
    for episode in range(episodes):
        if shutdown_flag.is_set():
            break
        
        farm_env.reset()
        terminated = False
        truncated = False
        rewards = []
        
        while not (terminated or truncated):
            action = farm_env.action_space.sample()
            _, reward, terminated, truncated, info = farm_env.step(action)
            rewards.append(reward)
        infos.append(info)
        pbar.update(1)
        pbar.set_postfix({
            "reward":np.mean(rewards),
            "harvest_reward": info['harvest_reward'],
                          })
        mean_rewards.append(np.mean(rewards))
        
    pbar.close()
    print(average_infos(infos))
    np.save(CONFIGS.rewards_path / f'random_{CONFIGS.experiment_name}.npy', mean_rewards)
    np.save(CONFIGS.infos_path / f'random_{CONFIGS.experiment_name}_infos.npy', infos)
    
def main():
    farm_size = (CONFIGS.farm_size, CONFIGS.farm_size)
    weekly_water_supply = farm_size[0] * farm_size[1] * CONFIGS.weekly_water_tile_supply  
    weekly_fertilizer_supply = farm_size[0] * farm_size[1] * CONFIGS.weekly_fertilizer_tile_supply  
    weekly_labour_supply = farm_size[0] * farm_size[1] * CONFIGS.weekly_labour_tile_supply
    farm_env = gym.envs.make("Farm-v0", 
                             years=CONFIGS.years, 
                             farm_size=farm_size, 
                             yearly_water_supply=weekly_water_supply*52, 
                             yearly_fertilizer_supply=weekly_fertilizer_supply*52, 
                             weekly_labour_supply=weekly_labour_supply,
                             exp=CONFIGS.yield_exponent)
    
    # save a json with the configs used in this training:
    
    
    sample_obs, _ = farm_env.reset()
    sample_action = farm_env.action_space.sample()
    
    total_cells = farm_size[0] * farm_size[1]
    state_dim = flatten_observation(sample_obs).shape[0] 
    action_dim = flatten_action(sample_action).shape[0] 
    
    print("Predicted Observation Space Shape:", (state_dim,))
    print("Observation Space Shape:", flatten_observation(sample_obs).shape)
    print("Sample Action Shape:", flatten_action(sample_action).shape)
 
    random_train(farm_env, 50)

    agent = PolicyGradientAgent(state_dim=state_dim,
                                action_dim=action_dim, 
                                total_cells=total_cells, 
                                learning_rate=CONFIGS.learning_rate, 
                                gamma=CONFIGS.gamma, 
                                device=CONFIGS.device, 
                                use_icm=CONFIGS.icm,
                                entropy_bonus = CONFIGS.entropy_bonus,
                                beta_max=CONFIGS.beta_start,
                                beta_min=CONFIGS.beta_end,
                                beta_decay=CONFIGS.beta_decay
                                )
    print(f"Agent n params: {sum(p.numel() for p in agent.policy.parameters() if p.requires_grad)}")
    epsilon = CONFIGS.eps_start
    
    pbar = tqdm(total=CONFIGS.episodes, desc="Training", unit="step")
    mean_rewards = []
    infos = []
    state, _ = farm_env.reset()
    for episode in range(CONFIGS.episodes):
        r_ints = []
        r_exts = []
        if shutdown_flag.is_set():
            break
        
        if sigusr1_flag.is_set():
            np.save(CONFIGS.infos_path / f'reinforce_{CONFIGS.experiment_name}_infos.npy', infos)
            print("saved. resuming training...")
            sigusr1_flag.clear()
        
        obs, _ = farm_env.reset(options={'episode': episode})
        state = flatten_observation(obs)
        terminated = False
        truncated = False
        rewards = []
        agent.episode = episode  # Update the current episode for beta cycle
        r_ints = []
        r_exts = [] 
        while not (terminated or truncated):
            if CONFIGS.eps_greedy and random.random() < epsilon:
                action = farm_env.action_space.sample()
                log_prob = torch.tensor([0.0], device=agent.device)
                entropy  = torch.tensor(0.0, device=agent.device)
            else:
                action, log_prob, entropy = agent.select_action_hybrid(state)
                
            next_state, reward, terminated, truncated, info = farm_env.step(action)
            next_state = flatten_observation(next_state)
            
            r_exts.append(reward)
            if agent.icm is not None:
                r_int = agent.icm.compute_intrinsic_reward(state, next_state, flatten_action(action))
                r_ints.append(CONFIGS.icm_eta * r_int)
                reward += CONFIGS.icm_eta * r_int
                agent.icm.update(state=state, next_state=next_state, action=flatten_action(action))
            
            agent.store_outcome(log_prob, reward, state, entropy)            
            state = next_state
            
            rewards.append(r_exts) # otherwise we would poison the icm plot with the intrinsic reward
        
        pbar.update(1)
        pbar.set_postfix({
            "tot_reward": np.mean(rewards),
            "int": np.mean(r_ints) if len(r_ints) > 0 else 0.0,
            "ext": np.mean(r_exts),
            "harvest": info['harvest_reward'],
        })
        mean_rewards.append(np.mean(rewards))
        infos.append(info)
        # Decay epsilon
        epsilon = max(epsilon * CONFIGS.eps_decay, CONFIGS.eps_end)
        
        # update policy every BATCH_SIZE episodes
        if (episode + 1) % CONFIGS.batch_size == 0:
            agent.update_policy()
            print(average_infos(infos[-CONFIGS.batch_size:]))
            
    # save infos and rewards array
    np.save(CONFIGS.infos_path / f'reinforce_{CONFIGS.experiment_name}_infos.npy', infos)
    
    pbar.close()
    print(mean_rewards)
    np.save(CONFIGS.rewards_path / f'reinforce_{CONFIGS.experiment_name}_rewards.npy', mean_rewards)
    
if __name__ == "__main__":
    main()

