import torch
import torch.nn as nn
import torch.optim as optim

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

