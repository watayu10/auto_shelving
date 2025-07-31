import glob
import math
import os
import pickle
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import torch
from tqdm import tqdm

from agents.agent import BC, BCWithEncoder
from agents.encoder import ResnetEncoder, pre_trained_ResnetEncoder
from config import bc_config, iq_learn_config
from envs.env import ProductShelvingEnv
from my_utils.transform import getPosEulerFromObs, quat_rotate_vector
from my_wrappers.iq_learn_wrapper import SACWrapper
from my_wrappers.pybullet_ur5_robotiq_wrapper import UR5Robotiq85
from third_party.pybullet_ur5_robotiq.utilities import Camera


def get_args(cfg):
    cfg.device = "cpu"
    return cfg


def model_control():

    init_obs = {
        "ee_pos": (0.10994562362094863, 0.009423244468561909, 0.5020576288266748),
        "ee_ori": (
            0.5006460351652267,
            0.5008130750189577,
            -0.49895446145809175,
            0.49958408376814817,
        ),
    }
    init_action = getPosEulerFromObs(init_obs) + [0.04]
    cfg1 = bc_config()
    args = get_args(cfg1)

    # set seeds
    device = torch.device(args.device)
    if device.type == "cuda" and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # models
    obs_dim = 512 * 3
    action_dim = 4
    action_range = [-2, 2]
    # iq_learn
    # agent = SACWrapper(obs_dim, action_dim, action_range, args.train.batch, args)
    # agent.load("results", "_iq")
    # bc
    agent = BCWithEncoder(obs_dim, action_dim, args)
    agent.load(
        agent_path="results/bc_agent50_frist", encoder_path="results/bc_encoder50_frist"
    )

    # encoder = pre_trained_ResnetEncoder()

    # env
    camera1 = Camera((0, -1, 1.25), (0, 0, 0), (0, 0, 1), 0.1, 5, (224, 224), 40)
    camera2 = Camera((0, 0, 0.5), (-0.40, 0, 0.25), (0, 0, 1), 0.1, 5, (224, 224), 40)
    robot = UR5Robotiq85((0, 0.5, 0), (0, 0, 0))
    env = ProductShelvingEnv(robot, vis=True)
    obs = env.reset()

    ee_images = []
    camera1_images = []
    camera2_images = []

    pre_action1 = []
    pre_action2 = []
    pre_action3 = []
    pre_action4 = []

    # env.SIMULATION_STEP_DELAY = 0
    for _ in range(224):
        print(_)
        """
        obs, reward, done, info = env.step(actions[count], "end")
        """
        # Mount a camera at the tip of the end effector
        # camera pos is current ee_pos + ee_local pos[0,0,-0.1]
        ee_pos = obs["ee_pos"]
        ori = obs["ee_ori"]
        camera_local_pos = [0, 0, -0.1]
        local_target = [1, 0, 0]
        local_up = [0, 0, -1]
        camera_world_pos = quat_rotate_vector(ori, camera_local_pos)
        world_target = quat_rotate_vector(ori, local_target)
        wordl_up = quat_rotate_vector(ori, local_up)
        camera_ee = Camera(
            (
                ee_pos[0] + camera_world_pos[0],
                ee_pos[1] + camera_world_pos[1],
                ee_pos[2] + camera_world_pos[2],
            ),
            (world_target[0], world_target[1], world_target[2]),
            (wordl_up[0], wordl_up[1], wordl_up[2]),
            0.1,
            5,
            (224, 224),
            40,
        )

        camera1_image, _, _ = camera1.shot()
        camera2_image, _, _ = camera2.shot()
        ee_image, _, _ = camera_ee.shot()
        ee_images.append(ee_image)
        camera1_images.append(camera1_image)
        camera2_images.append(camera2_image)
        obs = (
            np.stack(
                [
                    camera1_image,
                    camera2_image,
                    ee_image,
                ]
            )
            / 255
        ).astype(np.float32)
        obs = obs[:, np.newaxis, :, :, :]
        obs = torch.from_numpy(obs)
        obs = agent.encoder(obs)
        # action = agent.choose_action(obs, False).tolist()
        action1, action2, action3, action4 = agent.agent.action(obs)

        init_action[0] = action1
        init_action[1] = action2
        init_action[2] = action3
        init_action[-1] = action4

        # gripper_open_lenthの調整0から0.1の範囲に制限
        open_length = max(0, min(0.1, init_action[-1]))
        init_action[-1] = open_length
        pre_action1.append(action1.item())
        pre_action2.append(action2.item())
        pre_action3.append(action3.item())
        pre_action4.append(action4.item())

        obs, reward, done, info = env.step(init_action, "end")

        """
        count += 1
        print(count)
        """
        # time.sleep(1)
    with open("pre_action1.pkl", "wb") as f:
        pickle.dump(pre_action1, f)
    with open("pre_action2.pkl", "wb") as f:
        pickle.dump(pre_action2, f)
    with open("pre_action3.pkl", "wb") as f:
        pickle.dump(pre_action3, f)
    with open("pre_action4.pkl", "wb") as f:
        pickle.dump(pre_action4, f)

    with open("ee_images.pkl", "wb") as f:
        pickle.dump(ee_images, f)
    with open("camera1_images.pkl", "wb") as f:
        pickle.dump(camera1_images, f)
    with open("camera2_images.pkl", "wb") as f:
        pickle.dump(camera2_images, f)


def visualize_export_action():
    folder_path = "./expert/7"
    pickle_files = glob.glob(os.path.join(folder_path, "*.pkl"))
    loaded_dict = {}

    for file in pickle_files:
        with open(file, "rb") as f:
            loaded_dict[os.path.basename(file)] = pickle.load(f)

    for key in loaded_dict.keys():
        print(len(loaded_dict[key]))

    actions = loaded_dict["actions.pkl"]
    actions_len = len(actions)
    count = 0
    # env
    camera1 = Camera((0, -1, 1.25), (0, 0, 0), (0, 0, 1), 0.1, 5, (224, 224), 40)
    camera2 = Camera((0, 0, 0.5), (-0.40, 0, 0.25), (0, 0, 1), 0.1, 5, (224, 224), 40)
    robot = UR5Robotiq85((0, 0.5, 0), (0, 0, 0))
    env = ProductShelvingEnv(robot, vis=True)
    obs = env.reset()

    # env.SIMULATION_STEP_DELAY = 0
    while True:
        print(actions[count])
        obs, reward, done, info = env.step(actions[count], "end")
        count += 1
        if count == actions_len - 1:
            break


if __name__ == "__main__":
    model_control()
    # visualize_export_action()
