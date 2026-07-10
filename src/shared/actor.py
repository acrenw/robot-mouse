"""
actor network, shared between training and pi inference
obs: [dist_proxy, angle, visible, cat_vx, cat_vy, sensor_front, mouse_speed]
act: [v_raw, omega_raw] in [-1, 1], scaled by the caller

TODO: if inference is too slow on pi, export this to onnx
TODO: could try a bigger network (128 hidden) if the policy plateaus
"""

import torch
import torch.nn as nn

OBS_DIM = 7
ACT_DIM = 2
HIDDEN = 64 # keeping this small so it runs fast on pi


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        # arbitrary names, parent registers all internally
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, ACT_DIM)
        )
        # learnable param (wide -> narrow as policy gets confident and has less exploration)
        # optimizer updates things marked as nn.Parameter (weights are nn.Parameter too)
        self.log_std = nn.Parameter(torch.zeros(ACT_DIM)) # gives tensor of that dimension with zeros

    def forward(self, obs): # nn.Module calls self.forward(*args, **kwargs) when you call the class
        """
        pytorch convention
        """
        mu = self.net(obs) # mean of the action distribution
        std = self.log_std.exp().clamp(1e-3, 1.0) # exp so it's always positive, clamp so it doesn't go crazy
        return mu, std

    def get_action(self, obs):
        """
        inputs: obs is a batch of obs
        stochastic, used during training
        """
        mu, std = self(obs) # calls hooks then self.forward(), baked into nn.Module
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()  # rsample = reparameterized sample, lets gradients flow through
        action = torch.tanh(z)  # squash to [-1, 1] (still a batch, ie: (256, 2))

        # the -log(1 - action^2) term accounts for the change of variables from z to action
        # 1e-6 prevents log(0) when action is exactly +-1
        # sums probabilities of action space together
        action_log_prob = (dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return action, action_log_prob

    def get_deterministic_action(self, obs):
        """
        greedy, used at deploy time
        """
        mu, _ = self(obs)
        return torch.tanh(mu) # just take the mean, no sampling
