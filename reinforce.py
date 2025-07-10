import torch
import torch.nn as nn
import torch.optim as optim
import math
from typing import Dict, Any

BETA_MAX = 0.2
BETA_MIN = 0.01
BETA_T = 150  # Number of steps for one cycle of beta

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

class ICM(nn.Module):
    """
    Intrinsic Curiosity Module (ICM)
    Ref: Pathak et al. (2017)
    """
    def __init__(self, state_dim, action_dim, feature_dim=128, lr=1e-3, beta=0.2, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.device = device
        self.beta = beta  # weighting between forward & inverse losses
        # Feature encoder phi(s)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(inplace=True),
            nn.Linear(256, feature_dim),
            nn.ELU(inplace=True)
        )
        # Inverse model: phi(s), phi(s') -> predict action
        self.inverse = nn.Sequential(
            nn.Linear(2*feature_dim, 256),
            nn.ELU(inplace=True),
            nn.Linear(256, action_dim)
        )
        # Forward model: phi(s), action -> predict phi(s')
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ELU(inplace=True),
            nn.Linear(256, feature_dim)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, state, next_state, action):
        """
        Given raw states and action, compute the intrinsic reward
        """
        # state, next_state and action to tensors
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # encode state features
        phi_s  = self.encoder(state)
        phi_s2 = self.encoder(next_state)
        
        # inverse reconstruction loss (regression): what action do you think was taken to go from s to s'?
        inv_in     = torch.cat([phi_s, phi_s2], dim=-1)
        action_pred = self.inverse(inv_in)
        inv_loss   = self.mse(action_pred, action)
        
        # forward prediction loss: given s and action, what is the next state (its embedding)?
        fwd_in       = torch.cat([phi_s.detach(), action], dim=-1)
        phi_s2_pred  = self.forward_model(fwd_in)
        fwd_loss     = self.mse(phi_s2_pred, phi_s2.detach())
        # intrinsic reward: scaled forward error. If phi_s2_pred is not close to phi_s2, then the agent hasn't learned to predict the next state well, hence we make it curious
        r_int = 0.5 * (phi_s2_pred - phi_s2).pow(2).sum(dim=-1)
        return r_int, inv_loss, fwd_loss
    
    def compute_intrinsic_reward(self, state, next_state, action):
        """Returns intrinsic reward, detached from graph."""
        r_int, _, _ = self.forward(state, next_state, action)
        return r_int.detach().cpu().numpy()

    def update(self, state, next_state, action):
        """
        Takes batches of transitions and updates ICM parameters.
        Returns intrinsic rewards.
        """
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        r_int, inv_loss, fwd_loss = self.forward(state, next_state, action)
        # combined ICM loss, we use the beta found in the paper
        loss = (1 - self.beta) * inv_loss + self.beta * fwd_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return r_int.detach()



class HybridPolicyNetwork(nn.Module):
    def __init__(self, input_dim, total_cells, n_crops):
        super().__init__()
        self.total_cells = total_cells
        self.n_crops = n_crops
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
        self.crop_mask_head = nn.Linear(256, total_cells  * (self.n_crops + 1))  # 4 crop types + 1 for no crop selected, we multiply to one-hot encode the crop mask
        self.harvest_mask_head = nn.Linear(256, total_cells)  # binary harvest mask, 1 if harvest, 0 if not
        # self.crop_select_head = nn.Linear(256, 4)  # 4 crop types


    def forward(self, x):
        x = self.shared(x)
        
        # Continuous outputs
        water_mean = torch.sigmoid(self.water_head(x))
        fertilizer_mean = torch.sigmoid(self.fertilizer_head(x))
        water_std = torch.exp(self.water_log_std)
        fertilizer_std = torch.exp(self.fertilizer_log_std)
        
        # Discrete outputs
        crop_mask_logits = self.crop_mask_head(x) # take the logits
        harvest_mask_logits = self.harvest_mask_head(x)  # binary logits for harvest mask
        # crop_select_logits = self.crop_select_head(x) # also take the logits
        
        return water_mean, water_std, fertilizer_mean, fertilizer_std, crop_mask_logits, harvest_mask_logits

class PolicyGradientAgent:
    def __init__(self,
                 action_dim, 
                 state_dim, 
                 total_cells, 
                 n_crops=4,
                 learning_rate: float = 3e-4, 
                 gamma: float = .99, 
                 episodes: int = 1000,
                 use_icm: bool = True, 
                 device=torch.device("cuda") if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.total_cells = total_cells
        self.n_crops = n_crops
        self.policy = HybridPolicyNetwork(state_dim, self.total_cells, self.n_crops).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=20)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=episodes/batch_size, eta_min=1e-5)
        self.memory = []  # Stores (log_prob, reward, state)
        self.gamma = gamma
        self.episode = 0
        self.icm = ICM(state_dim=state_dim, action_dim=action_dim, device=self.device).to(self.device) if use_icm else None

    def select_action_hybrid(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        water_mean, water_std, fertilizer_mean, fertilizer_std, crop_mask_logits, harvest_mask_logits = self.policy(state)
        
        # Sample continuous actions from Gaussian policy
        water_dist = torch.distributions.Normal(water_mean, water_std)
        water_action = water_dist.sample()
        water_log_prob = water_dist.log_prob(water_action).sum(dim=-1)  # Compute log probability
                
        fertilizer_dist = torch.distributions.Normal(fertilizer_mean, fertilizer_std)
        fertilizer_action = fertilizer_dist.sample()
        fertilizer_log_prob = fertilizer_dist.log_prob(fertilizer_action).sum(dim=-1)  # Compute log probability
        
        # Sample discrete actions
        crop_mask_logits = crop_mask_logits.view(-1, self.total_cells, self.n_crops + 1)  # Reshape to (1, total_cells * num_classes)
        crop_mask_dist = torch.distributions.Categorical(logits=crop_mask_logits)
        crop_mask_action = crop_mask_dist.sample()
        crop_mask_log_prob = crop_mask_dist.log_prob(crop_mask_action).sum(dim=-1)  # Compute log probability
        
        harvest_mask_dist = torch.distributions.Bernoulli(logits=harvest_mask_logits)
        harvest_mask_action = harvest_mask_dist.sample()
        harvest_mask_log_prob = harvest_mask_dist.log_prob(harvest_mask_action).sum(dim=-1)  # Compute log probability

        # Combine actions and log probabilities
        action = {
            'water_amount': torch.clamp(water_action, 0, 1).detach().cpu().numpy().squeeze(),
            'fertilizer_amount': torch.clamp(fertilizer_action, 0, 1).detach().cpu().numpy().squeeze(),
            'crop_mask': crop_mask_action.squeeze(0).detach().cpu().numpy(),
            'harvest_mask': harvest_mask_action.squeeze(0).detach().cpu().numpy(),
        }
        log_prob = water_log_prob + fertilizer_log_prob + crop_mask_log_prob + harvest_mask_log_prob
        entropy = (
            water_dist.entropy().mean()
            + fertilizer_dist.entropy().mean()
            + crop_mask_dist.entropy().mean()
        ) / 4
        return action, log_prob, entropy
        
    def beta_cycle(self, T=150, beta_min=0.01, beta_max=0.2):
        t = self.episode % T
        cos_out = (1 + math.cos(math.pi * t / T)) / 2 # 1+ brings it to [0, 2], then divide by 2 to bring it to [0, 1]
        return beta_min + (beta_max - beta_min) * (cos_out) # it does half the cycle, so it goes from beta_max to beta_min slowly and back to beta_max instantly, like CosineAnnealingWithWarmRestarts
    
    def store_outcome(self, log_prob, reward, state, entropy):
        self.memory.append((log_prob, reward, state, entropy))

    def update_policy(self):
        # global EXPLORATION_COEFFICIENT
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

            # exploration_bonus = self.beta_cycle(T=BETA_T, beta_max=BETA_MAX, beta_min=BETA_MIN) * entropy
            # policy_loss.append(-log_prob * advantage - exploration_bonus)  # Gradient ascent
            policy_loss.append(-log_prob * advantage) # no exploration bonus for now

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        episode_reward = sum(reward for _, reward, _, _ in self.memory)
        self.scheduler.step(episode_reward)
        
        # Exploration coefficient decay
        # EXPLORATION_COEFFICIENT *= EXPLORATION_COEFFICIENT_DECAY
        # EXPLORATION_COEFFICIENT = max(EXPLORATION_COEFFICIENT, 0.01)
        
        # Clear memory after updating policy
        self.memory = []
