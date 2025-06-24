import environment as env
from reinforce import PolicyGradientAgent, PolicyNetwork
import numpy as np
import gymnasium as gym
from tqdm import tqdm

LEARNING_RATE = 3e-4
GAMMA = 0.99  # Discount factor
EPISODES = 3000  # Number of training episodes
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
        'crop_mask': action[:total_cells],
        # 'harvest_mask': np.zeros(total_cells, dtype=np.int8),
        'water_amount': action[total_cells:2*total_cells],
        'fertilizer_amount': action[2*total_cells:3*total_cells],
        'crop_selection': action[-1],  # Always plant wheat
    }


def main():
    years = 3
    farm_size = (3, 3)
    farm_env = gym.envs.make("Farm-v0", years=years, farm_size=farm_size, yearly_water_supply=100, yearly_fertilizer_supply=100, yearly_labor_supply=1000)
    
    sample_obs, _ = farm_env.reset()
    sample_action = farm_env.action_space.sample()
    
    total_cells = farm_size[0] * farm_size[1]
    state_dim = flatten_observation(sample_obs, years).shape[0] # total_cells * 6 + 5
    action_dim = flatten_action(sample_action).shape[0]  # total_cells * 3 + 1
    
    print("Predicted Observation Space Shape:", (state_dim,))
    print("Observation Space Shape:", flatten_observation(sample_obs, years).shape)
    
    print("Initial Observation:", sample_obs['farm_state'])
    
    print("Sample Action Shape:", flatten_action(sample_action).shape)
    print("Sample Action:", sample_action)
    
    # sample step
    
    # observation, reward, terminated, truncated, info = farm_env.step(sample_action)
    # print("Step Result:")
    # print("Crops:", observation['farm_state'][:, 0])
    # print("Growth:", observation['farm_state'][:, 1])
    # print("Health:", observation['farm_state'][:, 2])
    # print("Yield:", observation['farm_state'][:, 3])
    # print("Water:", observation['farm_state'][:, 4])
    # print("Fertilizer:", observation['farm_state'][:, 5])
    
    # print("Reward:", reward)
    # print("Terminated:", terminated)
    # print("Truncated:", truncated)
    # print("Info:", info)    
    

    agent = PolicyGradientAgent(state_dim=state_dim, action_dim=action_dim, learning_rate=LEARNING_RATE, gamma=GAMMA)
    
    
    pbar = tqdm(total=EPISODES, desc="Training", unit="step")
    mean_rewards = []
    water_reserve = []
    fertilizer_reserve = []
    state, _ = farm_env.reset()
    for episode in range(EPISODES):
        obs, _ = farm_env.reset()
        state = flatten_observation(obs, years)
        done = False
        truncated = False
        rewards = []
        
        while not (done or truncated):
            # action = farm_env.action_space.sample()
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, info = farm_env.step(agent_action_to_env(action=action, total_cells=total_cells))
            agent.store_outcome(log_prob, reward, state)
            state = flatten_observation(next_state, years)

            
            rewards.append(reward)
            water_reserve.append(sample_obs['water_supply'])
            fertilizer_reserve.append(sample_obs['fertilizer_supply'])
        
        
        
        pbar.update(1)
        pbar.set_postfix(reward=np.mean(rewards))
        mean_rewards.append(np.mean(rewards))
        
        # update policy every BATCH_SIZE episodes
        if (episode + 1) % BATCH_SIZE == 0:
            agent.update_policy()
    
    pbar.close()
    print(mean_rewards)
    
    print("Final Water Reserve:", np.mean(water_reserve))

    # print(fertilizer_reserve)
if __name__ == "__main__":
    main()

