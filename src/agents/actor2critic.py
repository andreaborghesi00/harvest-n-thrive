import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical, Bernoulli
from agents.icm import ICM

EXPLORATION_COEFFICIENT = 0.2
EXPLORATION_COEFFICIENT_DECAY = 0.99

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
    def forward(self, x):
        return x + self.block(x)
    
class HybridActorCritic(nn.Module):
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

        )
        
        self.policy_head = nn.Sequential(
            ResidualBlock(512),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            ResidualBlock(256),
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(inplace=True),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU(inplace=True),

            nn.Linear(64, 1)
            
        )
        
        # policy heads
        # Continuous heads
        # self.water_head = nn.Linear(256, total_cells)
        self.water_head = nn.Sequential(
            nn.Linear(256, total_cells)
        )
        self.fertilizer_head = nn.Sequential(
            nn.Linear(256, total_cells)
        )
        
        self.water_log_std = nn.Parameter(torch.zeros(total_cells))
        self.fertilizer_log_std = nn.Parameter(torch.zeros(total_cells))
        
        # Discrete heads
        # self.crop_mask_head = nn.Sequential(
        #     nn.Linear(256, total_cells  * (self.n_crops + 1)) 
        #     ) # 4 crop types + 1 for no crop selected, we multiply to one-hot encode the crop mask
        
        self.crop_mask_head = nn.Sequential(
            nn.Linear(256, total_cells)
        )        
        
        self.crop_type_head = nn.Sequential(
            nn.Linear(256, self.n_crops)
        )
        
        self.harvest_mask_head = nn.Sequential(
            nn.Linear(256, total_cells)
            )  # binary harvest mask, 1 if harvest, 0 if not
        
        # value head
        # self.value_head = nn.Linear(256, 1)
        
    def forward(self, x):
        h = self.shared(x)
        
        policy_features = self.policy_head(h)  # shared features for policy heads
        
        # policy dists
        water_mean = torch.sigmoid(self.water_head(policy_features))
        water_std = torch.exp(self.water_log_std)

        fertilizer_mean = torch.sigmoid(self.fertilizer_head(policy_features))
        fertilizer_std = torch.exp(self.fertilizer_log_std)
        
        crop_mask_logits = self.crop_mask_head(policy_features) # take the logits
        harvest_mask_logits = self.harvest_mask_head(policy_features)  # binary logits for harvest mask
        crop_select_logits = self.crop_type_head(policy_features) # also take the logits
        
        v = self.value_head(h).squeeze(-1)     # feed the "decisions" into the critic head
        return water_mean, water_std, fertilizer_mean, fertilizer_std, crop_mask_logits, harvest_mask_logits, crop_select_logits, v


class ActorCriticAgent:
    def __init__(self,
                 state_dim, 
                 action_dim,
                 total_cells, 
                 n_crops=4,
                 learning_rate: float = 3e-4, 
                 gamma: float = .99, 
                 c1: float = 0.5,
                 use_icm: bool = False,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device
        self.a2c = HybridActorCritic(state_dim, total_cells, n_crops).to(self.device)
        self.optimizer = optim.Adam(self.a2c.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=20, verbose=True)
        self.memory = []  # Stores (log_prob, reward, state)
        self.gamma = gamma
        self.c1 = c1  # Coefficient for critic loss
        icm = ICM(state_dim=state_dim, action_dim=action_dim, device=self.device).to(self.device) if use_icm else None
        
    def select_action_hybrid(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        water_mean, water_std, fertilizer_mean, fertilizer_std, crop_mask_logits, harvest_mask_logits, crop_select_logits, v = self.a2c(state)
        
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
        
        harvest_mask_dist = torch.distributions.Bernoulli(logits=harvest_mask_logits)
        harvest_mask_action = harvest_mask_dist.sample()
        harvest_mask_log_prob = harvest_mask_dist.log_prob(harvest_mask_action).sum(dim=-1)  # Compute log probability

        # Combine actions and log probabilities
        action = {
            'water_amount': torch.clamp(water_action, 0, 1).detach().cpu().numpy().squeeze(),
            'fertilizer_amount': torch.clamp(fertilizer_action, 0, 1).detach().cpu().numpy().squeeze(),
            'crop_mask': crop_mask_action.squeeze(0).detach().cpu().numpy(),
            'harvest_mask': harvest_mask_action.squeeze(0).detach().cpu().numpy(),
            'crop_type': crop_select_action.item()
        }
        log_prob = water_log_prob + fertilizer_log_prob + crop_mask_log_prob + harvest_mask_log_prob + crop_select_log_prob
        entropy = (
            water_dist.entropy().mean()
            + fertilizer_dist.entropy().mean()
            + crop_mask_dist.entropy().mean()
            + harvest_mask_dist.entropy().mean()
            + crop_select_dist.entropy()
        ) / 5
        return action, log_prob, entropy, v
        
    def store_outcome(self, log_prob, reward, v, next_state, entropy):
        next_v =  self.a2c(torch.FloatTensor(next_state).unsqueeze(0).to(self.device))[-1] if next_state is not None else 0.0
        self.memory.append({
            'logp': log_prob,
            'entropy': entropy,
            'reward': torch.tensor(reward, device=self.device),
            'value': v.detach(),
            'next_value': torch.tensor(next_v, device=self.device) # Get next value estimate
            })

    def update(self):
        global EXPLORATION_COEFFICIENT
        R = 0  # Discounted return

        logps      = torch.stack([e['logp']      for e in self.memory])
        entropies  = torch.stack([e['entropy']   for e in self.memory])
        rewards    = torch.tensor([e['reward']    for e in self.memory], device=self.device)
        values     = torch.tensor([e['value']     for e in self.memory], device=self.device)
        next_vals  = torch.tensor([e['next_value']for e in self.memory], device=self.device)

        # td(lambda): compute the eligibility traces
        # R = 0
        # deltas = []
        # for r, v, next_v in zip(rewards.flip(0), values.flip(0), next_vals.flip(0)):
        #     R = r + self.gamma * next_v - v
        #     deltas.append(R)
        # deltas = torch.tensor(deltas[::-1], device=self.device)  # Reverse to match original order

        deltas = rewards + self.gamma * next_vals - values

        actor_loss = -(logps * deltas.detach()).mean() - EXPLORATION_COEFFICIENT * entropies.mean()
        critic_loss = deltas.pow(2).mean()
        loss = actor_loss + self.c1 * critic_loss
        
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.a2c.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        episode_reward = sum(e['reward'] for e in self.memory)
        self.scheduler.step(episode_reward)
        
        # Exploration coefficient decay
        # EXPLORATION_COEFFICIENT *= EXPLORATION_COEFFICIENT_DECAY
        # EXPLORATION_COEFFICIENT = max(EXPLORATION_COEFFICIENT, 0.01)
        
        # Clear memory after updating policy
        self.memory.clear()
