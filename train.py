import environment as env
from reinforce import PolicyGradientAgent
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import torch

LEARNING_RATE = 3e-4
GAMMA = 0.99  # Discount factor
EPISODES = 1000  # Number of training episodes
BATCH_SIZE = 10  # Update policy after X episodes
HIDDEN_UNITS = 256  # Reduced hidden layer size for efficiency

def flatten_observation(obs, years):
    return np.concatenate([
        obs['farm_state'].flatten(),
        obs['water_supply'],
        obs['fertilizer_supply'],
        obs['labor_supply'],
        [obs['current_week'] / 52],
        [obs['current_year'] / years]
    ])
    
def flatten_action(action):
    return np.concatenate([
        action['crop_mask'].flatten(),
        [action['crop_selection']],
        # action['harvest_mask'].flatten(),
        action['water_amount'].flatten(),
        action['fertilizer_amount'].flatten()
    ])
    

def agent_action_to_env(action, total_cells):
    return {
        'crop_mask': (action[:total_cells] > 0).astype(int),  # Threshold at 0
        'crop_selection': np.clip(np.round(action[2*total_cells]), 0, 3).astype(int),
        'water_amount': (action[total_cells:2*total_cells] + 1) / 2,  # Scale to [0,1]
        'fertilizer_amount': (action[2*total_cells:3*total_cells] + 1) / 2
    }

def random_train(farm_env, years, episodes):
    pbar = tqdm(total=episodes, desc="Training", unit="step")
    mean_rewards = []
    water_reserve = []
    fertilizer_reserve = []
    state, _ = farm_env.reset()
    for episode in range(episodes):
        obs, _ = farm_env.reset()
        state = flatten_observation(obs, years)
        terminated = False
        truncated = False
        rewards = []
        
        while not (terminated or truncated):
            action = farm_env.action_space.sample()
            # action, log_prob = agent.select_action_hybrid(state)
            # next_state, reward, terminated, truncated, info = farm_env.step(agent_action_to_env(action=action, total_cells=total_cells))
            next_state, reward, terminated, truncated, info = farm_env.step(action)
            # agent.store_outcome(log_prob, reward, state)
            state = flatten_observation(next_state, years)
            rewards.append(reward)

        pbar.update(1)
        pbar.set_postfix(reward=np.mean(rewards))
        mean_rewards.append(np.mean(rewards))
        
    pbar.close()
    print(mean_rewards)
    plt.figure(figsize=(12, 6))
    plt.plot(mean_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Mean Reward')
    plt.title('Random mean rewards')
    plt.grid()
    plt.ylim(-3, 5)
    plt.savefig('random_mean_rewards.png')
    
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
    farm_env = gym.envs.make("Farm-v0", years=years, farm_size=farm_size, yearly_water_supply=9*52, yearly_fertilizer_supply=5*52, yearly_labor_supply=1000)
    
    sample_obs, _ = farm_env.reset()
    sample_action = farm_env.action_space.sample()
    
    total_cells = farm_size[0] * farm_size[1]
    state_dim = flatten_observation(sample_obs, years).shape[0] # total_cells * 6 + 5
    
    print("Predicted Observation Space Shape:", (state_dim,))
    print("Observation Space Shape:", flatten_observation(sample_obs, years).shape)
    print("Sample Action Shape:", flatten_action(sample_action).shape)
 
    # random_train(farm_env, years, 50)

    agent = PolicyGradientAgent(state_dim=state_dim, total_cells=total_cells, learning_rate=LEARNING_RATE, gamma=GAMMA)
    
    eps_start   = 1.0      # start fully random
    eps_end     = 0.1      # end with some randomness
    eps_decay   = 0.995    # decay per episode

    epsilon = eps_start
    
    pbar = tqdm(total=EPISODES, desc="Training", unit="step")
    mean_rewards = []
    water_reserve = []
    fertilizer_reserve = []
    state, _ = farm_env.reset()
    for episode in range(EPISODES):
        obs, _ = farm_env.reset()
        state = flatten_observation(obs, years)
        terminated = False
        truncated = False
        rewards = []
        
        while not (terminated or truncated):
            # action = farm_env.action_space.sample()
            
            if random.random() < epsilon:
            # if False:
                # exploration
                action = farm_env.action_space.sample()
                log_prob = torch.tensor([0.0], device=agent.device)
                entropy  = torch.tensor(0.0, device=agent.device)
            else:
                # exploitation
                action, log_prob, entropy = agent.select_action_hybrid(state)
                
            # next_state, reward, terminated, truncated, info = farm_env.step(agent_action_to_env(action=action, total_cells=total_cells))
            next_state, reward, terminated, truncated, info = farm_env.step(action)
            agent.store_outcome(log_prob, reward, state, entropy)
            state = flatten_observation(next_state, years)
            
            rewards.append(reward)
        
        pbar.update(1)
        pbar.set_postfix(reward=np.mean(rewards))
        mean_rewards.append(np.mean(rewards))
        
        # Decay epsilon
        epsilon = max(epsilon * eps_decay, eps_end)
        
        # update policy every BATCH_SIZE episodes
        if (episode + 1) % BATCH_SIZE == 0:
            agent.update_policy()
        
        if episode % 10 == 0:
            print(f"Action stats: "
                f"Water={action['water_amount'].mean():.2f}±{action['water_amount'].std():.2f} "
                f"Fert={action['fertilizer_amount'].mean():.2f} "
                f"CropMask={action['crop_mask'].mean():.2f} "
                f"CropSel={action['crop_selection']}")
            
    pbar.close()
    print(mean_rewards)
    np.save('reinforce_mean_rewards.npy', mean_rewards)
    
    plt.figure(figsize=(12, 6))
    plt.plot(mean_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Mean Reward')
    plt.title('REINFORCE mean rewards')
    plt.grid()
    plt.ylim(-3, 5)
    plt.savefig('reinforce_mean_rewards.png')
    
    # print(fertilizer_reserve)
if __name__ == "__main__":
    main()

