import os
import pickle
from glob import glob

import numpy as np
import torch


class Memory:
    """
    exportフォルダに1から番号のついたフォルダがありその中に
    actions.pkl,camera1_images.pkl,camera2_images.pkl,ee_images.pkl
    があることを前提にしている
    """

    def __init__(
        self,
        data_path="C:/Users/Watan/Desktop/Code/auto_shelving/expert",
        # action_discretization=False,
    ):
        base_dir = data_path
        subdirs = sorted([d for d in os.listdir(base_dir) if d.isdigit()], key=int)
        self.buffer = {
            "actions": [],
            "camera1_images": [],
            "camera2_images": [],
            "ee_images": [],
            "done": [],
        }
        for d in subdirs:
            folder_path = os.path.join(base_dir, d)
            pkl_files = glob(os.path.join(folder_path, "*.pkl"))
            temp_len = 0
            # index = None
            for file in pkl_files:
                if file:
                    if os.path.basename(file) == "actions.pkl":
                        with open(file, "rb") as f:
                            data = pickle.load(f)
                        """action = np.append(
                            np.zeros((1, 7)),
                            np.diff(np.array(data), axis=0),
                            axis=0,
                        )
                        index = np.where(~np.all(action == 0, axis=1))
                        action = action[index]"""
                        self.buffer["actions"] += data
                        temp_len = len(data)
                    else:
                        with open(file, "rb") as f:
                            data = pickle.load(f)

                        self.buffer[os.path.basename(file)[:-4]] += data
                        """
                        temp = np.array(data).astype(np.float16)  # [index]
                        self.buffer[os.path.basename(file)[:-4]].append(temp)
                        temp_len = temp.shape[0]"""

            self.buffer["done"] += [0] * (temp_len - 1) + [1]

        for key in self.buffer.keys():
            # self.buffer[key] = np.concatenate(self.buffer[key], axis=0)
            self.buffer[key] = np.array(self.buffer[key])

        """
        # actionが-0.04から0.01づつ0.04までの値を取るように離散化してそれを0から8の番号に変換
        if action_discretization:
            self.buffer["actions"] = (
                (np.round(self.buffer["actions"], decimals=2) + 0.04) / 0.01
            ).astype(int)
        """

        # 負のaction 0,actionが0の時1,正のaction 2
        """
        if action_discretization:
            rounded_action = np.round(self.buffer["actions"], decimals=2)
            action = np.where(
                rounded_action < 0, 0, np.where(rounded_action > 0, 2, 1)
            ).astype(int)
            self.buffer["actions"] = action
        """
        self.size = min([self.buffer[key].shape[0] for key in self.buffer.keys()])

        """
        print(self.buffer["actions"].shape)
        print(self.buffer["camera1_images"].shape)
        print(self.buffer["camera2_images"].shape)
        print(self.buffer["ee_images"].shape)
        print(self.buffer["done"].shape)
        """

    def get_samples(self, batch_size, device="cpu"):
        samples = np.random.choice(
            np.arange(1, self.size - 1), batch_size, replace=False
        )
        samples_isDone = self.buffer["done"][samples]
        samples[samples_isDone] -= 1

        previous_samples = samples - 1
        next_samples = samples + 1
        obs = (
            np.stack(
                [
                    self.buffer["camera1_images"][previous_samples],
                    self.buffer["camera2_images"][previous_samples],
                    self.buffer["ee_images"][previous_samples],
                ]
            )
            / 255
        ).astype(np.float32)
        actions = self.buffer["actions"][samples].astype(np.float32)
        """
        next_obs = (
            np.stack(
                [
                    self.buffer["camera1_images"][next_samples],
                    self.buffer["camera2_images"][next_samples],
                    self.buffer["ee_images"][next_samples],
                ]
            )
            / 255
        ).astype(np.float32)
        """
        # done = self.buffer["done"][samples].reshape(-1, 1).astype(np.float32)
        # reward = np.zeros((batch_size, 1)).astype(np.float32)

        obs = torch.from_numpy(obs).to(device)
        # next_obs = torch.from_numpy(next_obs).to(device)
        actions = torch.from_numpy(actions).to(
            device
        )  # 離散アクションの場合long()が必要
        # reward = torch.from_numpy(reward)
        # done = torch.from_numpy(done).to(device)

        return (
            obs,
            actions,
        )  # (obs, next_obs, actions, reward, done)


"""
memory = Memory("expert/")  # action_discretization=True)
print(memory.buffer["actions"].shape)
print(memory.buffer["camera1_images"].shape)
print(memory.buffer["camera2_images"].shape)
print(memory.buffer["ee_images"].shape)
batch = memory.get_samples(10)
print(batch[1])"""
"""
for i in range(7):
    print(i, np.unique(memory.buffer["actions"][:, i]))
print("--------------------------------------")
for i in range(2):
    print(batch[i].shape)
print("--------------------------------------")
print(batch[1][:, [0, 1, 2, -1]].shape)
"""
