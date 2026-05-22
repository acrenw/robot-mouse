"""
actor network, shared between training and pi inference
obs: [dist_proxy, angle, visible, cat_vx, cat_vy, sensor_front, mouse_speed]
act: [v_raw, omega_raw] in [-1, 1], scaled by the caller

# TODO: if inference is too slow on pi, export this to onnx
# TODO: could try a bigger network (128 hidden) if the policy plateaus
"""

import torch
import torch.nn as nn

OBS_DIM = 7
ACT_DIM = 2
HIDDEN = 64 # keeping this small so it runs fast on pi


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
        )
        self.mu_head = nn.Linear(HIDDEN, ACT_DIM)
        self.log_std = nn.Parameter(torch.zeros(ACT_DIM))

    def forward(self, obs):
        x = self.net(obs)
        mu = self.mu_head(x) # mean of the action distribution
        std = self.log_std.exp().clamp(1e-3, 1.0) # exp so it's always positive, clamp so it doesn't go crazy
        return mu, std

    def get_action(self, obs):
        """
        stochastic, used during training
        """
        mu, std = self(obs)
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()  # rsample = reparameterized sample, lets gradients flow through
        action = torch.tanh(z)  # squash to [-1, 1]
        # log prob of the action, corrected for the tanh squashing (standard SAC formula)
        # the -log(1 - action^2) term accounts for the change of variables from z to action
        # 1e-6 prevents log(0) when action is exactly +-1
        log_prob = (dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return action, log_prob

    def get_deterministic_action(self, obs):
        """
        greedy, used at deploy time
        """
        mu, _ = self(obs)
        return torch.tanh(mu) # just take the mean, no sampling
