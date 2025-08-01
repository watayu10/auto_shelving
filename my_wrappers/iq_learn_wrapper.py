import os
import sys

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

sys.path.append(os.path.abspath("third_party/IQ_Learn/iq_learn"))

from agent.sac import SAC
from agent.sac_models import DiagGaussianActor, DoubleQCritic, orthogonal_init_
from utils.utils import soft_update


# loggerを削除
class SACWrapper(SAC):
    def __init__(self, obs_dim, action_dim, action_range, batch_size, args):
        self.gamma = args.gamma
        self.batch_size = batch_size
        self.action_range = action_range
        self.device = torch.device(args.device)
        self.args = args
        agent_cfg = args.agent

        self.critic_tau = agent_cfg.critic_tau
        self.learn_temp = agent_cfg.learn_temp
        self.actor_update_frequency = agent_cfg.actor_update_frequency
        self.critic_target_update_frequency = agent_cfg.critic_target_update_frequency

        self.critic = DoubleQCriticWrapper(obs_dim, action_dim, args).to(self.device)

        self.critic_target = DoubleQCriticWrapper(obs_dim, action_dim, args).to(
            self.device
        )
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = DiagGaussianActorWrapper(obs_dim, action_dim, [-5, 2]).to(
            self.device
        )

        self.log_alpha = torch.tensor(np.log(agent_cfg.init_temp)).to(self.device)
        self.log_alpha.requires_grad = True
        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = Adam(
            self.actor.parameters(), lr=agent_cfg.actor_lr, betas=agent_cfg.actor_betas
        )
        self.critic_optimizer = Adam(
            self.critic.parameters(),
            lr=agent_cfg.critic_lr,
            betas=agent_cfg.critic_betas,
        )
        self.log_alpha_optimizer = Adam(
            [self.log_alpha], lr=agent_cfg.alpha_lr, betas=agent_cfg.alpha_betas
        )
        self.train()
        self.critic_target.train()

    def choose_action(self, state, sample=False):
        dist = self.actor(state)
        action = dist.sample() if sample else dist.mean
        # assert action.ndim == 2 and action.shape[0] == 1
        return action.detach()

    def update(self, replay_buffer, step):
        obs, next_obs, action, reward, done = replay_buffer.get_samples(
            self.batch_size, self.device
        )

        losses = self.update_critic(obs, action, reward, next_obs, done, step)

        if step % self.actor_update_frequency == 0:
            actor_alpha_losses = self.update_actor_and_alpha(obs, step)
            losses.update(actor_alpha_losses)

        if step % self.critic_target_update_frequency == 0:
            soft_update(self.critic, self.critic_target, self.critic_tau)

        return losses

    def update_critic(self, obs, action, reward, next_obs, done, step):

        with torch.no_grad():
            next_action, log_prob, _ = self.actor.sample(next_obs)

            target_Q = self.critic_target(next_obs, next_action)
            target_V = target_Q - self.alpha.detach() * log_prob
            target_Q = reward + (1 - done) * self.gamma * target_V

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action, both=True)
        q1_loss = F.mse_loss(current_Q1, target_Q)
        q2_loss = F.mse_loss(current_Q2, target_Q)
        critic_loss = q1_loss + q2_loss

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # self.critic.log(logger, step)
        return {
            "critic_loss/critic_1": q1_loss.item(),
            "critic_loss/critic_2": q2_loss.item(),
            "loss/critic": critic_loss.item(),
        }

    def update_actor_and_alpha(self, obs, step):
        action, log_prob, _ = self.actor.sample(obs)
        actor_Q = self.critic(obs, action)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        losses = {
            "loss/actor": actor_loss.item(),
            "actor_loss/target_entropy": self.target_entropy,
            "actor_loss/entropy": -log_prob.mean().item(),
        }

        # self.actor.log(logger, step)
        if self.learn_temp:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (
                self.alpha * (-log_prob - self.target_entropy).detach()
            ).mean()

            alpha_loss.backward()
            self.log_alpha_optimizer.step()

            losses.update(
                {
                    "alpha_loss/loss": alpha_loss.item(),
                    "alpha_loss/value": self.alpha.item(),
                }
            )
        return losses


class DoubleQCriticWrapper(DoubleQCritic, nn.Module):
    def __init__(self, obs_dim, action_dim, args):
        nn.Module.__init__(self)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.args = args

        # Q1 architecture
        self.Q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
        )

        # Q2 architecture
        self.Q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
        )

        self.apply(orthogonal_init_)


class DiagGaussianActorWrapper(DiagGaussianActor, nn.Module):
    def __init__(self, obs_dim, action_dim, log_std_bounds):
        nn.Module.__init__(self)
        self.log_std_bounds = log_std_bounds
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, action_dim * 2),
        )

        self.outputs = dict()
        self.apply(orthogonal_init_)
