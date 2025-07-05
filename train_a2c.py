import environment as env
from reinforce import PolicyGradientAgent
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import torch
from actor2critic import HybridActorCritic, ActorCriticAgent

LEARNING_RATE = 3e-4
GAMMA = 0.99  # Discount factor
EPISODES = 1000  # Number of training episodes
BATCH_SIZE = 10  # Update policy after X episodes
HIDDEN_UNITS = 256  # Reduced hidden layer size for efficiency
C1 = 0.5  # Coefficient for critic loss
YEARS = 10
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def flatten_observation(obs):
    return np.concatenate([
        obs['farm_state'].flatten(),
        obs['water_supply'],
        obs['fertilizer_supply'],
        obs['labor_supply'],
        [obs['current_week'] / 52],
        [obs['current_year'] / YEARS]
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


def random_train(farm_env, episodes):
    pbar = tqdm(total=episodes, desc="Training", unit="step")
    mean_rewards = []
    water_reserve = []
    fertilizer_reserve = []
    state, _ = farm_env.reset()
    for episode in range(episodes):
        obs, _ = farm_env.reset()
        state = flatten_observation(obs)
        terminated = False
        truncated = False
        rewards = []
        
        while not (terminated or truncated):
            action = farm_env.action_space.sample()
            # action, log_prob = agent.select_action_hybrid(state)
            # next_state, reward, terminated, truncated, info = farm_env.step(agent_action_to_env(action=action, total_cells=total_cells))
            next_state, reward, terminated, truncated, info = farm_env.step(action)
            # agent.store_outcome(log_prob, reward, state)
            state = flatten_observation(next_state)
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


def main():
    farm_size = (3, 3)
    total_cells = farm_size[0] * farm_size[1]
    farm_env = gym.envs.make("Farm-v0", years=YEARS, farm_size=farm_size, yearly_water_supply=9*52, yearly_fertilizer_supply=5*52, yearly_labor_supply=1000)
    
    sample_obs, _ = farm_env.reset()
    sample_action = farm_env.action_space.sample()
    
    state_dim = flatten_observation(sample_obs).shape[0] # total_cells * 6 + 5
    
    print("Predicted Observation Space Shape:", (state_dim,))
    print("Observation Space Shape:", flatten_observation(sample_obs).shape)
    print("Sample Action Shape:", flatten_action(sample_action).shape)
 
    # random_train(farm_env, years, 50)

    agent = ActorCriticAgent(state_dim=state_dim, total_cells=total_cells, learning_rate=LEARNING_RATE, gamma=GAMMA, c1=C1, device=DEVICE)
    
    pbar = tqdm(total=EPISODES, desc="Training", unit="step")
    mean_rewards = []
    state, _ = farm_env.reset()

    # main loop
    for episode in range(EPISODES):
        obs, _ = farm_env.reset()
        state = flatten_observation(obs)
        terminated = False
        truncated = False
        rewards = []
        
        # single-episode loop
        while not (terminated or truncated):
            action, log_prob, entropy, v = agent.select_action_hybrid(state)
                
            next_state, reward, terminated, truncated, _ = farm_env.step(action)
            agent.store_outcome(log_prob=log_prob, reward=reward, next_state=flatten_observation(next_state) if not (terminated or truncated) else None, v=v, entropy=entropy)
            state = flatten_observation(next_state)
            
            rewards.append(reward)
        
        pbar.update(1)
        pbar.set_postfix(reward=np.mean(rewards))
        mean_rewards.append(np.mean(rewards))
        
        # update policy every BATCH_SIZE episodes
        if (episode + 1) % BATCH_SIZE == 0:
            agent.update()
        
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

