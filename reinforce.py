import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2

EXPLORATION_COEFFICIENT = 0.2
EXPLORATION_COEFFICIENT_DECAY = 0.99

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
    def forward(self, x):
        return x + self.block(x)

# class HybridPolicyNetwork(nn.Module):
#     def __init__(self, input_dim, total_cells):
#         super().__init__()
#         self.total_cells = total_cells
#         self.shared =nn.Sequential(
#             nn.Linear(input_dim, 256),
#             nn.LayerNorm(256),
#             nn.SiLU(inplace=True),

#             nn.Linear(256, 512),
#             nn.LayerNorm(512),
#             nn.SiLU(inplace=True),

#             nn.Linear(512, 1024),
#             nn.LayerNorm(1024),
#             nn.SiLU(inplace=True),
#             ResidualBlock(1024),
#             nn.Linear(1024, 512),
#             nn.LayerNorm(512),
#             nn.SiLU(inplace=True),

#             nn.Linear(512, 256),
#             nn.LayerNorm(256),
#             nn.SiLU(inplace=True),
#             ResidualBlock(256),
#         )
        
#         # Continuous heads
#         self.water_head = nn.Linear(256, total_cells)
#         self.fertilizer_head = nn.Linear(256, total_cells)
        
#         self.water_log_std = nn.Parameter(torch.zeros(total_cells))
#         self.fertilizer_log_std = nn.Parameter(torch.zeros(total_cells))
        
#         # Discrete heads
#         self.crop_mask_head = nn.Linear(256, total_cells)
#         self.crop_select_head = nn.Linear(256, 4)  # 4 crop types


class HybridPolicyNetwork(nn.Module):
    def __init__(self, input_dim, total_cells):
        super().__init__()
        self.total_cells = total_cells
        self.shared =nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),

            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.SiLU(inplace=True),

            ResidualBlock(512),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            ResidualBlock(256),
        )
        
        # Continuous heads
        self.water_head = nn.Linear(256, total_cells)
        self.fertilizer_head = nn.Linear(256, total_cells)
        
        self.water_log_std = nn.Parameter(torch.zeros(total_cells))
        self.fertilizer_log_std = nn.Parameter(torch.zeros(total_cells))
        
        # Discrete heads
        self.crop_mask_head = nn.Linear(256, total_cells)
        self.crop_select_head = nn.Linear(256, 4)  # 4 crop types


    def forward(self, x):
        x = self.shared(x)
        
        # Continuous outputs
        water_mean = torch.sigmoid(self.water_head(x))
        fertilizer_mean = torch.sigmoid(self.fertilizer_head(x))
        water_std = torch.exp(self.water_log_std)
        fertilizer_std = torch.exp(self.fertilizer_log_std)
        
        # Discrete outputs
        crop_mask_logits = self.crop_mask_head(x) # take the logits
        crop_select_logits = self.crop_select_head(x) # also take the logits
        
        return water_mean, water_std, fertilizer_mean, fertilizer_std, crop_mask_logits, crop_select_logits

class PolicyGradientAgent:
    def __init__(self, state_dim, total_cells, learning_rate: float = 3e-4, gamma: float = .99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = HybridPolicyNetwork(state_dim, total_cells).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=20, verbose=True)
        self.memory = []  # Stores (log_prob, reward, state)
        self.gamma = gamma

    def select_action_hybrid(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        water_mean, water_std, fertilizer_mean, fertilizer_std, crop_mask_logits, crop_select_logits = self.policy(state)
        
        # Sample continuous actions from Gaussian policy
        water_dist = torch.distributions.Normal(water_mean, water_std)
        water_action = water_dist.sample()
        water_log_prob = water_dist.log_prob(water_action).sum(dim=-1)  # Compute log probability
                
        fertilizer_dist = torch.distributions.Normal(fertilizer_mean, fertilizer_std)
        fertilizer_action = fertilizer_dist.sample()
        fertilizer_log_prob = fertilizer_dist.log_prob(fertilizer_action).sum(dim=-1)  # Compute log probability
        
        # Sample discrete actions
        crop_mask_dist = torch.distributions.Bernoulli(logits=crop_mask_logits)
        crop_mask_action = crop_mask_dist.sample()
        crop_mask_log_prob = crop_mask_dist.log_prob(crop_mask_action).sum(dim=-1)  # Compute log probability
        
        crop_select_dist = torch.distributions.Categorical(logits=crop_select_logits)
        crop_select_action = crop_select_dist.sample()
        crop_select_log_prob = crop_select_dist.log_prob(crop_select_action)  # Compute log probability
        
        # Combine actions and log probabilities
        action = {
            'water_amount': torch.clamp(water_action, 0, 1).detach().cpu().numpy().squeeze(),
            'fertilizer_amount': torch.clamp(fertilizer_action, 0, 1).detach().cpu().numpy().squeeze(),
            'crop_mask': crop_mask_action.squeeze(0).detach().cpu().numpy(),
            'crop_selection': crop_select_action.item()
        }
        log_prob = water_log_prob + fertilizer_log_prob + crop_mask_log_prob + crop_select_log_prob
        entropy = (
            water_dist.entropy().mean()
            + fertilizer_dist.entropy().mean()
            + crop_mask_dist.entropy().mean()
            + crop_select_dist.entropy().mean()
        ) / 4
        return action, log_prob, entropy
        
    def store_outcome(self, log_prob, reward, state, entropy):
        self.memory.append((log_prob, reward, state, entropy))

    def update_policy(self):
        global EXPLORATION_COEFFICIENT
        R = 0  # Discounted return
        policy_loss = []
        returns = []

        # Compute discounted rewards
        for _, reward, _, _ in reversed(self.memory):
            R = reward + self.gamma * R
            returns.insert(0, R) # head insertion so that we get the correct order

        returns = torch.tensor(returns).to(self.device)

        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        baseline = sum(returns) / len(returns)

        # Compute policy gradient loss with advantage function
        for (log_prob, _, _, entropy), R in zip(self.memory, returns):
            advantage = R - baseline  

            exploration_bonus = EXPLORATION_COEFFICIENT * entropy

            policy_loss.append(-log_prob * advantage - exploration_bonus)  # Gradient ascent

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        episode_reward = sum(reward for _, reward, _, _ in self.memory)
        self.scheduler.step(episode_reward)
        
        # Exploration coefficient decay
        EXPLORATION_COEFFICIENT *= EXPLORATION_COEFFICIENT_DECAY
        EXPLORATION_COEFFICIENT = max(EXPLORATION_COEFFICIENT, 0.01)
        
        # Clear memory after updating policy
        self.memory = []
