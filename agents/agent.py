import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from agents.encoder import ResnetEncoder, pre_trained_ResnetEncoder
from my_wrappers.iq_learn_wrapper import SACWrapper


class SACWithEncoder:
    def __init__(
        self,
        obs_dim,
        action_dim,
        action_range,
        args,
    ):
        self.pre_trained_encoder = args.pre_trained_encoder
        self.agent = SACWrapper(
            obs_dim, action_dim, action_range, args.train.batch, args
        )
        if self.pre_trained_encoder:
            self.encoder = pre_trained_ResnetEncoder().to(args.device)
            self.encoder_optimizer = None
        else:
            self.encoder = encoder = ResnetEncoder().to(args.device)
            self.encoder_optimizer = AdamW(
                encoder.parameters(), lr=1e-4, weight_decay=1e-3
            )

    def train_encoder(self):
        if not self.pre_trained_encoder:
            self.encoder.train()

    def eval(self):
        self.agent.train(False)
        if not self.pre_trained_encoder:
            self.encoder.eval()


class BCWithEncoder:
    def __init__(self, obs_dim, action_dim, args):
        self.pre_trained_encoder = args.pre_trained_encoder
        self.agent = BC(obs_dim, action_dim, args.device).to(args.device)
        if self.pre_trained_encoder:
            self.encoder = pre_trained_ResnetEncoder().to(args.device)
            self.encoder_optimizer = None
        else:
            self.encoder = encoder = ResnetEncoder().to(args.device)
            self.encoder_optimizer = AdamW(
                encoder.parameters(), lr=1e-4, weight_decay=1e-3
            )

    def save(self, path="result"):
        torch.save(self.agent.state_dict(), "results/bc_agent")
        if not self.pre_trained_encoder:
            torch.save(self.encoder.state_dict(), "results/bc_encoder")

    def load(self, agent_path, encoder_path=None):
        self.agent.load_state_dict(torch.load(agent_path))
        if not self.pre_trained_encoder:
            self.encoder.load_state_dict(torch.load(encoder_path))

    def train(self, training=True):
        self.agent.train(training)
        if not self.pre_trained_encoder:
            self.encoder.train()

    def eval(self):
        self.agent.train(False)
        if not self.pre_trained_encoder:
            self.encoder.eval()


class BC(nn.Module):
    def __init__(self, obs_dim, action_dim, device):
        super().__init__()
        self.action_dim = action_dim
        self.layers = nn.Sequential(
            nn.Linear(obs_dim, obs_dim),  # obs_dim 1536
            nn.ReLU(),
            # nn.Linear(obs_dim, obs_dim),
            # nn.ReLU(),
            # nn.Linear(obs_dim, obs_dim),
            # nn.ReLU(),
            # nn.Linear(obs_dim, obs_dim),
            # nn.ReLU(),
            # nn.Linear(obs_dim, obs_dim),
            # nn.ReLU(),
            nn.Linear(obs_dim, obs_dim),
            nn.ReLU(),
        )
        self.head1 = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            # nn.Linear(512, 512),
            # nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.head2 = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            # nn.Linear(512, 512),
            # nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.head3 = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            # nn.Linear(512, 512),
            # nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.head4 = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            # nn.Linear(512, 512),
            # nn.ReLU(),
            nn.Linear(512, 1),
        )

        """
        self.action_table = torch.arange(3) * 0.01 - 0.01
        self.loss_weights = (
            torch.tensor(
                [
                    [1.111, 0.766, 1.124],
                    [1.399, 0.084, 1.517],
                    [1.326, 0.344, 1.33],
                    [1.56, 0.068, 1.373],
                ]
            )
            .float()
            .to(device)
        )  # expert dataに存在するactionの割合の逆数
        self.crossEntropyLoss_list = [
            nn.CrossEntropyLoss(weight=self.loss_weights[0]),
            nn.CrossEntropyLoss(weight=self.loss_weights[1]),
            nn.CrossEntropyLoss(weight=self.loss_weights[2]),
            nn.CrossEntropyLoss(weight=self.loss_weights[3]),
        ]"""
        self.optimizer = AdamW(self.layers.parameters(), lr=1e-4, weight_decay=1e-3)

    def forward(self, x):
        out = self.layers(x)
        out1 = self.head1(out)
        out2 = self.head2(out)
        out3 = self.head3(out)
        out4 = self.head4(out)
        return out1, out2, out3, out4  # batch,action_dim,action_table_dim

    def loss(self, x, expert_action):
        out1, out2, out3, out4 = self.forward(x)
        loss = 0
        loss += F.mse_loss(out1, expert_action[:, 0].unsqueeze(1))
        loss += F.mse_loss(out2, expert_action[:, 1].unsqueeze(1))
        loss += F.mse_loss(out3, expert_action[:, 2].unsqueeze(1))
        loss += F.mse_loss(out4, expert_action[:, 3].unsqueeze(1))
        """
        loss += self.crossEntropyLoss_list[0](out1, expert_action[:, 0])
        loss += self.crossEntropyLoss_list[1](out2, expert_action[:, 1])
        loss += self.crossEntropyLoss_list[2](out3, expert_action[:, 2])
        loss += self.crossEntropyLoss_list[3](out4, expert_action[:, 3])
        """
        return loss

    def action(self, x):
        out1, out2, out3, out4 = self.forward(x)

        """
        out1_index = torch.argmax(out1, dim=1)
        out2_index = torch.argmax(out2, dim=1)
        out3_index = torch.argmax(out3, dim=1)
        out4_index = torch.argmax(out4, dim=1)
        action = self.action_table[
            torch.concatenate([out1_index, out2_index, out3_index, out4_index], dim=0)
        ]"""
        return out1, out2, out3, out4  # out1_index, out2_index, out3_index, out4_index


class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.linear1 = nn.Linear(obs_dim, 256)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(256, action_dim)

    def _forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class MulitQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.action_dim = action_dim
        self.linear = nn.Linear(obs_dim, 512)
        self.relu = nn.ReLU()
        q_nets = [QNetwork(512, action_dim) for _ in range(action_dim)]
        self.q_nets = nn.Sequential(q_nets)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        outputs = []
        for i in range(self.action_dim):
            outputs.append(self.q_nets[i](x))
        return outputs
