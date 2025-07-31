import glob
import os
import pickle
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np

folder_path = "./logs/BC"
pickle_files = glob.glob(os.path.join(folder_path, "*.pkl"))
loaded_dict = {}

for file in pickle_files:
    with open(file, "rb") as f:
        loaded_dict[os.path.basename(file)[:-4]] = pickle.load(f)

x = [
    i
    for i in range(
        len(loaded_dict["loss_frist"])
        # + len(loaded_dict["loss_132_231"])
        # + len(loaded_dict["loss_51_131"])
        # + len(loaded_dict["loss_0_50"])
    )
]
print(len(x))
y = (
    # loaded_dict["loss_0_50"]
    # + loaded_dict["loss_51_131"]
    # + loaded_dict["loss_132_231"]
    loaded_dict["loss_frist"]
)
plt.plot(x, y)
plt.ylim([0, 0.025])
plt.show()
"""
x = [i for i in range(len(loaded_dict["loss"]))]

for key in loaded_dict.keys():
    if (
        key == "actor_loss_entropy"
        or key == "actor_loss_target_entropy"
        or key == "v0"
        or key == "loss_actor"
        # or key == "chi2_loss"
    ):
        continue
    plt.plot(x, loaded_dict[key], label=key)
plt.legend()
plt.show()
"""
