from __future__ import annotations

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

import gymnasium as gym


class A2C(nn.Module):
    """
    (Synchronous) Advantage Actor-Critic (A2C) agent class.  
    """

    def __init__(self, 
                 n_features: int,
                 n_actions: int,
                 device: torch.device,
                 critic_lr: float,
                 actor_lr: float,
                 n_envs: int) -> None:
        """Initialize the actor and critic networks and their optimizers.
        ### Parameters
        - `n_features`: int, number of features in the state
        - `n_actions`: int, number of actions
        - `device`: torch.device, device to use for computation
        - `critic_lr`: float, learning rate for the critic
        - `actor_lr`: float, learning rate for the actor
        - `n_envs`: int, number of parallel environments to use"""
        super().__init__()
        self.device = device
        self.n_envs = n_envs

        critic_layers = [
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32, 1), # estimate V(s)
        ]

        actor_layers = [
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(
                32, n_actions
            ),  # estimate action logits (will be fed into a softmax later)
        ]
        # define actor and critic networks
        self.critic:nn.Sequential = nn.Sequential(*critic_layers).to(self.device)
        self.actor:nn.Sequential = nn.Sequential(*actor_layers).to(self.device)

        # define optimizer for actor and critic
        self.critic_optim = optim.RMSprop(self.critic.parameters(), lr=critic_lr)
        self.actor_optim = optim.RMSprop(self.actor.parameters(), lr=actor_lr)
    
    def forward(self, x:np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the actor and critic networks.

        ### Parameters
        - `x`: A batched vector of states, [n_env, state_dim]

        ### Return
        - `state_values`: torch.Tensor, estimated state values, with shape [n_env, ]
        - `action_logits_vec`: torch.Tensor, estimated action logits, with shape [n_env, n_actions]
        """
        x = torch.Tensor(x).to(self.device)
        state_values = self.critic(x)
        action_logits_vec = self.actor(x)
        return (state_values, action_logits_vec)

    def select_action(self, x: np.ndarray) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """
        Return a tuple the chosen actions and the log-probs of the actions.

        ### Return
         - a tuple: (actions, action_log_probs, state_values, entropy)
         Four torch.Tensor objects with shape [n_env, ]
        """
        state_values, action_logits_vec = self.forward(x)
        action_pd = torch.distributions.Categorical(logits=action_logits_vec)
        actions = action_pd.sample()
        action_log_probs = action_pd.log_prob(actions)
        entropy = action_pd.entropy()
        return (actions, action_log_probs, state_values, entropy)
    
    def get_losses(self, 
                   rewards:torch.Tensor, 
                   action_log_probs: torch.Tensor,
                   value_preds: torch.Tensor,
                   entropy: torch.Tensor,
                   masks: torch.Tensor,
                   gamma: float,
                   lam: float,
                   ent_coeft: float,
                   device: torch.device
                   )-> tuple[torch.Tensor, torch.Tensor]:
        """
        ### Parameters
        - `rewards`: torch.Tensor, with the rewards for each time step in the episode, with shape `[n_steps_per_update, n_env]`
        - `action_log_probs`: torch.Tensor, with the log-probabilities of the actions taken, with shape `[n_steps_per_update, n_env]`
        - `value_preds`: torch.Tensor, with the state values prediction for each time step in the episode, with shape `[n_steps_per_update, n_env]`
        - `masks`: torch.Tensor
        - `gamma`: float, discount factor
        - `lam`: float, GAE hyperparameter

        ### Return
        - A tuple, (critic_loss, actor_loss)
        - critic_loss: torch.Tensor, with shape [1, ]
        - actor_loss: torch.Tensor, with shape [1, ]
        """
        T = len(rewards) # the n_steps_per_update
        advantages = torch.zeros_like(T, self.n_envs, device=device)
        # []
        # compute the advanteges using GAE
        gae = 0.0
        for t in reversed(range(T-1)):
            td_error = (-value_preds[t] + rewards[t] + gamma * masks[t]* value_preds[t+1])
            gae = td_error + gamma * lam * masks[t] * gae
            advantages[t] = gae

        # calculate the loss of the minibatch for actor and critic
        critic_loss = advantages.pow(2).mean()

        # give a bonus for higher entropy, to encourage exploration
        actor_loss = -(advantages.detach() * action_log_probs).mean() - ent_coeft * entropy.mean()
        return (critic_loss, actor_loss)
    
    def update_parameters(self, 
                          critic_loss: torch.Tensor,
                          actor_loss: torch.Tensor)->None:
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

n_envs = 3
n_updates = 1000
n_steps_per_update = 128
randomize_domain = False

# agent hyperparams
gamma = 0.999
lam = 0.95
ent_coeft = 0.01
actor_lr = 0.001
critic_lr = 0.005

if __name__ == "__main__":
    # envs = gym.vector.make("LunarLander-v2", num_envs=3, max_episode_steps=600)
   if randomize_domain:
    envs = gym.vector.AsyncVectorEnv(
        [
            lambda: gym.make(
                "LunarLander-v2",
                gravity=np.clip(
                    np.random.normal(loc=-10.0, scale=1.0), a_min=-11.99, a_max=-0.01
                ),
                enable_wind=np.random.choice([True, False]),
                wind_power=np.clip(
                    np.random.normal(loc=15.0, scale=1.0), a_min=0.01, a_max=19.99
                ),
                turbulence_power=np.clip(
                    np.random.normal(loc=1.5, scale=0.5), a_min=0.01, a_max=1.99
                ),
                max_episode_steps=600,
            )
            for i in range(n_envs)
        ]
        )
   else:
    envs = gym.vector.make("LunarLander-v2", num_envs=n_envs, max_episode_steps=600)

    obs_shape = envs.single_action_space.shape[0]
    action_shape = envs.single_action_space.n

    use_cuda = False
    if use_cuda:
       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
       device = torch.device("cpu")
    
    # init the agent
    agent = A2C(obs_shape, action_shape, device, critic_lr, actor_lr, n_envs)
