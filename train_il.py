import os
import pickle
import random
import types

import keyboard
import numpy as np
import torch

from agents.agent import BCWithEncoder, SACWithEncoder
from config import bc_config, iq_learn_config
from memory import Memory
from third_party.IQ_Learn.iq_learn.iq import iq_loss
from third_party.IQ_Learn.iq_learn.utils.utils import (
    average_dicts,
    hard_update,
    soft_update,
)

"""
def get_args(cfg: DictConfig):
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.hydra_base_dir = os.getcwd()
    print(OmegaConf.to_yaml(cfg))
    return cfg
"""


def get_args(cfg):
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return cfg


# train iq_learn
def train_iq():
    cfg1 = iq_learn_config()
    args = get_args(cfg1)

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    obs_dim = 512 * 3
    action_dim = 4
    action_range = [-0.03, 0.03]
    agent = SACWithEncoder(
        obs_dim,
        action_dim,
        action_range,
        args,
    )
    expert_memory_replay = Memory()

    epochs = 100
    learn_steps = 0

    total_loss = []
    softq_loss = []
    value_loss = []
    v0 = []
    chi2_loss = []
    loss_actor = []
    actor_loss_target_entropy = []
    actor_loss_entropy = []

    agent.train_encoder()

    for epoch in range(epochs):
        print("epoch ==========", epoch)
        total_loss_temp = []
        softq_loss_temp = []
        value_loss_temp = []
        v0_temp = []
        chi2_loss_temp = []
        loss_actor_temp = []
        actor_loss_target_entropy_temp = []
        actor_loss_entropy_temp = []
        for _ in range(int(expert_memory_replay.size / args.train.batch)):
            agent.agent.iq_update = types.MethodType(iq_update, agent.agent)
            agent.agent.iq_update_critic = types.MethodType(
                iq_update_critic, agent.agent
            )
            losses = agent.agent.iq_update(
                expert_memory_replay,
                learn_steps,
                agent.encoder,
                agent.encoder_optimizer,
            )
            learn_steps += 1
            total_loss_temp.append(losses["total_loss"])
            softq_loss_temp.append(losses["softq_loss"])
            value_loss_temp.append(losses["value_loss"])
            v0_temp.append(losses["v0"])
            chi2_loss_temp.append(losses["chi2_loss"])
            loss_actor_temp.append(losses["loss/actor"])
            actor_loss_target_entropy_temp.append(losses["actor_loss/target_entropy"])
            actor_loss_entropy_temp.append(losses["actor_loss/entropy"])
        total_loss.append(sum(total_loss_temp) / len(total_loss_temp))
        softq_loss.append(sum(softq_loss_temp) / len(softq_loss_temp))
        value_loss.append(sum(value_loss_temp) / len(value_loss_temp))
        v0.append(sum(v0_temp) / len(v0_temp))
        chi2_loss.append(sum(chi2_loss_temp) / len(chi2_loss_temp))
        loss_actor.append(sum(loss_actor_temp) / len(loss_actor_temp))
        actor_loss_target_entropy.append(
            sum(actor_loss_target_entropy_temp) / len(actor_loss_target_entropy_temp)
        )
        actor_loss_entropy.append(
            sum(actor_loss_entropy_temp) / len(actor_loss_entropy_temp)
        )
        print("total_loss", total_loss[epoch])
        print("softq_loss", softq_loss[epoch])
        print("value_loss", value_loss[epoch])

        if keyboard.is_pressed("esc"):
            logs = {
                "total_loss": total_loss,
                "softq_loss": softq_loss,
                "value_loss": value_loss,
                "v0": v0,
                "chi2_loss": chi2_loss,
                "loss_actor": loss_actor,
                "actor_loss_target_entropy": actor_loss_target_entropy,
                "actor_loss_entropy": actor_loss_entropy,
            }
            for key in logs.keys():
                with open("logs/" + key + ".pkl", "wb") as f:
                    pickle.dump(logs[key], f)
            save(agent.agent, agent.encoder, args)
            break

    logs = {
        "total_loss": total_loss,
        "softq_loss": softq_loss,
        "value_loss": value_loss,
        "v0": v0,
        "chi2_loss": chi2_loss,
        "loss_actor": loss_actor,
        "actor_loss_target_entropy": actor_loss_target_entropy,
        "actor_loss_entropy": actor_loss_entropy,
    }
    for key in logs.keys():
        with open("logs/" + key + ".pkl", "wb") as f:
            pickle.dump(logs[key], f)
    save(agent.agent, agent.encoder, args)


def save(agent, encoder, args, output_dir="results"):
    if args.method.type == "sqil":
        name = f"sqil"
    else:
        name = f"iq"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    agent.save(f"{output_dir}/{args.agent.name}_{name}")
    if not args.pre_trained_encoder:
        torch.save(
            encoder.state_dict(), f"{output_dir}/{args.agent.name}_{name}_encoder"
        )


# loggerを削除,expert dataのみ,train_iq.pyを参考
def iq_update_critic(self, expert_batch, step, encoder, encoder_optimizer):
    args = self.args

    expert_obs, expert_next_obs, expert_action, expert_reward, expert_done = (
        expert_batch
    )

    # batch = get_concat_samples(policy_batch, expert_batch, args)
    batch = (
        expert_obs,
        expert_next_obs,
        expert_action,
        expert_reward,
        expert_done,
        torch.ones_like(expert_reward, dtype=torch.bool),
    )

    obs, next_obs, action = batch[0:3]

    agent = self
    current_V = self.getV(obs)
    if args.train.use_target:
        with torch.no_grad():
            next_V = self.get_targetV(next_obs)
    else:
        next_V = self.getV(next_obs)

    if "DoubleQ" in self.args.q_net._target_:
        current_Q1, current_Q2 = self.critic(obs, action, both=True)
        q1_loss, loss_dict1 = iq_loss(agent, current_Q1, current_V, next_V, batch)
        q2_loss, loss_dict2 = iq_loss(agent, current_Q2, current_V, next_V, batch)
        critic_loss = 1 / 2 * (q1_loss + q2_loss)
        # merge loss dicts
        loss_dict = average_dicts(loss_dict1, loss_dict2)
    else:
        current_Q = self.critic(obs, action)
        critic_loss, loss_dict = iq_loss(agent, current_Q, current_V, next_V, batch)

    # Optimize the critic
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    # step critic
    self.critic_optimizer.step()

    # to learn encoder
    if not args.pre_trained_encoder:
        encoder_optimizer.zero_grad()
        encoder_optimizer.step()

    return loss_dict


def iq_update(self, expert_buffer, step, encoder, encoder_optimizer):
    expert_batch = expert_buffer.get_samples(self.batch_size, self.device)
    encoded_obs = encoder(expert_batch[0])
    encoded_next_obs = encoder(expert_batch[1])
    # policy_action = self.choose_action(encoded_obs)

    expert_batch = (
        encoded_obs,
        encoded_next_obs,
        expert_batch[2],  # policy_action,
        expert_batch[3],
        expert_batch[4],
    )

    losses = self.iq_update_critic(expert_batch, step, encoder, encoder_optimizer)

    if self.actor and step % self.actor_update_frequency == 0:
        if not self.args.agent.vdice_actor:

            if self.args.offline:
                obs = expert_batch[0].detach()

            if self.args.num_actor_updates:
                for i in range(self.args.num_actor_updates):
                    actor_alpha_losses = self.update_actor_and_alpha(obs, step)

            losses.update(actor_alpha_losses)

    if step % self.critic_target_update_frequency == 0:
        if self.args.train.soft_update:
            soft_update(self.critic_net, self.critic_target_net, self.critic_tau)
        else:
            hard_update(self.critic_net, self.critic_target_net)
    return losses


def train_bc():
    cfg = bc_config()
    args = get_args(cfg)
    device = torch.device(args.device)
    if device.type == "cuda" and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    obs_dim = 512 * 3
    action_dim = 4
    agent = BCWithEncoder(obs_dim, action_dim, args)
    # agent.load(agent_path="results/bc_agent", encoder_path="results/bc_encoder")
    agent.train()
    expert_memory_replay = Memory()

    epochs = 50
    loss_list = []

    for epoch in range(epochs):
        print("epoch ==========", epoch)
        loss_list_temp = []
        for _ in range(int(expert_memory_replay.size / args.train.batch)):
            expert_batch = expert_memory_replay.get_samples(
                args.train.batch, args.device
            )
            encoded_obs = agent.encoder(expert_batch[0])
            loss = agent.agent.loss(encoded_obs, expert_batch[1][:, [0, 1, 2, -1]])
            agent.agent.optimizer.zero_grad()
            loss.backward()
            agent.agent.optimizer.step()

            # to learn encoder
            if not args.pre_trained_encoder:
                agent.encoder_optimizer.zero_grad()
                agent.encoder_optimizer.step()

            loss_list_temp.append(loss.item())

        loss_list.append(sum(loss_list_temp) / len(loss_list_temp))
        print("loss", loss_list[epoch])
        if epoch % 50 == 0 and epoch != 0:
            torch.save(agent.agent.state_dict(), "results/bc_agent" + str(epoch))
            with open("logs/BC/loss.pkl", "wb") as f:
                pickle.dump(loss_list, f)
            if not args.pre_trained_encoder:
                torch.save(
                    agent.encoder.state_dict(), "results/bc_encoder" + str(epoch)
                )

    with open("logs/BC/loss.pkl", "wb") as f:
        pickle.dump(loss_list, f)
    torch.save(agent.agent.state_dict(), "results/bc_agent")
    if not args.pre_trained_encoder:
        torch.save(agent.encoder.state_dict(), "results/bc_encoder")


if __name__ == "__main__":
    # train_iq()
    train_bc()
