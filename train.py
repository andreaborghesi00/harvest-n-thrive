import environment as env
from reinforce import PolicyGradientAgent
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import torch
import signal
import threading
from pathlib import Path

LEARNING_RATE = 3e-4
GAMMA = 0.99  # Discount factor
EPISODES = 2000  # Number of training episodes
BATCH_SIZE = 10  # Update policy after X episodes
HIDDEN_UNITS = 256  # Reduced hidden layer size for efficiency
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPERIMENT_NAME = "testing_2"

EPS_GREEDY = False  # Use epsilon-greedy exploration
ICM = False  # Use Intrinsic Curiosity Module
ENTROPY_BONUS = False

# For ICM
ETA = 130.0  # Weighting between internal curiosity and external reward

# Epsilon-greedy
EPS_START   = 1.0 # start fully random
EPS_END     = 0.1 # end with some randomness
EPS_DECAY   = 0.999 # decay per episode

RESULTS_DIR = Path("results/")
INFO_DIR = RESULTS_DIR / "infos"
REWARDS_DIR = RESULTS_DIR / "rewards"

INFO_DIR.mkdir(parents=True, exist_ok=True)
REWARDS_DIR.mkdir(parents=True, exist_ok=True)

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


def flatten_observation(obs, years):
    return np.concatenate([
        obs['farm_state'].flatten(),
        obs['water_supply'],
        obs['fertilizer_supply'],
        obs['labour_supply'],
        [obs['current_week'] / 52],
        [obs['current_year'] / years]
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


def random_train(farm_env, years, episodes):
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
    print(mean_rewards)
    print(average_infos(infos))
    np.save(REWARDS_DIR / f'random_{EXPERIMENT_NAME}.npy', mean_rewards)
    np.save(INFO_DIR / f'random_{EXPERIMENT_NAME}_infos.npy', infos)
    
def main():
    years = 10
    farm_size = (10, 10)
    weekly_water_supply = farm_size[0] * farm_size[1] * 0.75  # 0.5 water per cell per week
    weekly_fertilizer_supply = farm_size[0] * farm_size[1] * 0.5  # 0.5 fertilizer per cell per week
    weekly_labour_supply = 13 * farm_size[0] * farm_size[1]  # labour per cell per week
    farm_env = gym.envs.make("Farm-v0", years=years, farm_size=farm_size, yearly_water_supply=weekly_water_supply*52, yearly_fertilizer_supply=weekly_fertilizer_supply*52, weekly_labour_supply=weekly_labour_supply)
    
    sample_obs, _ = farm_env.reset()
    sample_action = farm_env.action_space.sample()
    
    total_cells = farm_size[0] * farm_size[1]
    state_dim = flatten_observation(sample_obs, years).shape[0] 
    action_dim = flatten_action(sample_action).shape[0] 
    
    print("Predicted Observation Space Shape:", (state_dim,))
    print("Observation Space Shape:", flatten_observation(sample_obs, years).shape)
    print("Sample Action Shape:", flatten_action(sample_action).shape)
 
    random_train(farm_env, years, 50)

    agent = PolicyGradientAgent(state_dim=state_dim,
                                action_dim=action_dim, 
                                total_cells=total_cells, 
                                learning_rate=LEARNING_RATE, 
                                gamma=GAMMA, 
                                device=DEVICE, 
                                use_icm=ICM,
                                entropy_bonus = ENTROPY_BONUS,
                                )
    print(f"Agent n params: {sum(p.numel() for p in agent.policy.parameters() if p.requires_grad)}")
    epsilon = EPS_START
    
    pbar = tqdm(total=EPISODES, desc="Training", unit="step")
    mean_rewards = []
    infos = []
    state, _ = farm_env.reset()
    for episode in range(EPISODES):
        r_ints = []
        r_exts = []
        if shutdown_flag.is_set():
            break
        
        if sigusr1_flag.is_set():
            np.save(INFO_DIR / f'reinforce_{EXPERIMENT_NAME}_infos.npy', infos)
            print("saved. resuming training...")
            sigusr1_flag.clear()
        
        obs, _ = farm_env.reset(options={'episode': episode})
        state = flatten_observation(obs, years)
        terminated = False
        truncated = False
        rewards = []
        agent.episode = episode  # Update the current episode for beta cycle
        r_ints = []
        r_exts = [] 
        while not (terminated or truncated):
            if EPS_GREEDY and random.random() < epsilon:
                action = farm_env.action_space.sample()
                log_prob = torch.tensor([0.0], device=agent.device)
                entropy  = torch.tensor(0.0, device=agent.device)
            else:
                action, log_prob, entropy = agent.select_action_hybrid(state)
                
            next_state, reward, terminated, truncated, info = farm_env.step(action)
            next_state = flatten_observation(next_state, years)
            
            r_exts.append(reward)
            if agent.icm is not None:
                r_int = agent.icm.compute_intrinsic_reward(state, next_state, flatten_action(action))
                r_ints.append(ETA * r_int)
                reward += ETA * r_int
                agent.icm.update(state=state, next_state=next_state, action=flatten_action(action))
            
            agent.store_outcome(log_prob, reward, state, entropy)            
            state = next_state
            
            rewards.append(reward)
        
        pbar.update(1)
        pbar.set_postfix({
            "tot_reward": np.mean(rewards),
            "int": np.mean(r_ints) if len(r_ints) > 0 else 0.0,
            "ext": np.mean(r_exts),
            "havest": info['harvest_reward'],
        })
        mean_rewards.append(np.mean(rewards))
        infos.append(info)
        # Decay epsilon
        epsilon = max(epsilon * EPS_DECAY, EPS_END)
        
        # update policy every BATCH_SIZE episodes
        if (episode + 1) % BATCH_SIZE == 0:
            agent.update_policy()
            print(average_infos(infos[-BATCH_SIZE:]))
            
    # save infos and rewards array
    np.save(INFO_DIR / f'reinforce_{EXPERIMENT_NAME}_infos.npy', infos)
    
    pbar.close()
    print(mean_rewards)
    np.save(REWARDS_DIR / f'reinforce_{EXPERIMENT_NAME}_rewards.npy', mean_rewards)
    
if __name__ == "__main__":
    main()

