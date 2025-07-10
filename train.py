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

LEARNING_RATE = 3e-4
GAMMA = 0.99  # Discount factor
EPISODES = 10000  # Number of training episodes
BATCH_SIZE = 10  # Update policy after X episodes
HIDDEN_UNITS = 256  # Reduced hidden layer size for efficiency
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPERIMENT_NAME = "testing"
# For ICM
ETA = 4.0  # Weighting between internal curiosity and external reward

shutdown_flag = threading.Event()

def on_sigint(signum, frame):
    print("\nSIGINT received; will stop after this episode.")
    shutdown_flag.set()
    
signal.signal(signal.SIGINT, on_sigint)
        # if episode % BATCH_SIZE == 0:
        #     print(f"Action stats: "
        #         f"Water={action['water_amount'].mean():.2f}±{action['water_amount'].std():.2f} "
        #         f"Fert={action['fertilizer_amount'].mean():.2f} "
        #         f"CropMask={action['crop_mask'].mean():.2f} "
        #         f"CropSel={action['crop_selection']}")


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
        # [action['crop_selection']],
        # action['harvest_mask'].flatten(),
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
    np.save(f'random_{EXPERIMENT_NAME}.npy', mean_rewards)
    
def sample_step_info(farm_env, years):
    sample_obs, _ = farm_env.reset()
    sample_action = farm_env.action_space.sample()
    observation, reward, terminated, truncated, info = farm_env.step(sample_action)
    state_dim = flatten_observation(sample_obs, years).shape[0] # total_cells * 6 + 5

    print("Predicted Observation Space Shape:", (state_dim,))
    print("Observation Space Shape:", flatten_observation(sample_obs, years).shape)
    
    print("Initial Observation:", sample_obs['farm_state'])
    
    print("Sample Action Shape:", flatten_action(sample_action).shape)
    print("Sample Action:", sample_action)
    
    print("Step Result:")
    print("Crops:", observation['farm_state'][:, 0])
    print("Growth:", observation['farm_state'][:, 1])
    print("Health:", observation['farm_state'][:, 2])
    print("Yield:", observation['farm_state'][:, 3])
    print("Water:", observation['farm_state'][:, 4])
    print("Fertilizer:", observation['farm_state'][:, 5])
    
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Info:", info)    

def main():
    years = 10
    farm_size = (3, 3)
    weekly_water_supply = farm_size[0] * farm_size[1] * 0.75  # 0.5 water per cell per week
    weekly_fertilizer_supply = farm_size[0] * farm_size[1] * 0.5  # 0.5 fertilizer per cell per week
    weekly_labour_supply = 1.75 * farm_size[0] * farm_size[1]  # labour per cell per week
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
                                use_icm=False)
    
    eps_start   = 1.0      # start fully random
    eps_end     = 0.1      # end with some randomness
    eps_decay   = 0.995    # decay per episode

    epsilon = eps_start
    
    pbar = tqdm(total=EPISODES, desc="Training", unit="step")
    mean_rewards = []
    infos = []
    state, _ = farm_env.reset()
    for episode in range(EPISODES):
        r_ints = []
        r_exts = []
        if shutdown_flag.is_set():
            break
        
        obs, _ = farm_env.reset(options={'episode': episode})
        state = flatten_observation(obs, years)
        terminated = False
        truncated = False
        rewards = []
        agent.episode = episode  # Update the current episode for beta cycle
        r_ints = []
        r_exts = [] 
        while not (terminated or truncated):
            
            # if random.random() < epsilon:
            if False:
                # exploration
                action = farm_env.action_space.sample()
                log_prob = torch.tensor([0.0], device=agent.device)
                entropy  = torch.tensor(0.0, device=agent.device)
            else:
                # exploitation
                action, log_prob, entropy = agent.select_action_hybrid(state)
                
            next_state, reward, terminated, truncated, info = farm_env.step(action)
            next_state = flatten_observation(next_state, years)
            
            r_exts.append(reward)
            if agent.icm is not None:
                r_int = agent.icm.compute_intrinsic_reward(state, next_state, flatten_action(action))
                r_ints.append(r_int)
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
        epsilon = max(epsilon * eps_decay, eps_end)
        
        # update policy every BATCH_SIZE episodes
        if (episode + 1) % BATCH_SIZE == 0:
            agent.update_policy()
            print(average_infos(infos[-BATCH_SIZE:]))
            
    # save infos and rewards array
    np.save(f'reinforce_{EXPERIMENT_NAME}_infos.npy', infos)
    
    pbar.close()
    print(mean_rewards)
    np.save(f'reinforce_{EXPERIMENT_NAME}_rewards.npy', mean_rewards)
    
if __name__ == "__main__":
    main()

