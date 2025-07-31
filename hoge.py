import glob
import os
import pickle

from agents.agent import BCWithEncoder
from config import bc_config
from memory import Memory


def get_args(cfg):
    cfg.device = "cpu"
    return cfg


cfg1 = bc_config()
args = get_args(cfg1)
obs_dim = 512 * 3
action_dim = 4
agent = BCWithEncoder(obs_dim, action_dim, args)
agent.load(agent_path="results/bc_agent", encoder_path="results/bc_encoder")

m = Memory("expert/", action_discretization=True)
