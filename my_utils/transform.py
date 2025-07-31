import numpy as np
import pybullet as p


def quat_rotate_vector(quat, v):  # local vector to world coordination
    qx, qy, qz, qw = quat
    v = np.array(v)
    t = 2 * np.cross([qx, qy, qz], v)
    return v + qw * t + np.cross([qx, qy, qz], t)


def getPosEulerFromObs(obs):
    pos = [0.0] * 3
    for i in range(len(obs["ee_pos"])):
        pos[i] = obs["ee_pos"][i]
    euler = p.getEulerFromQuaternion(obs["ee_ori"])
    posEuler = pos + list(euler)
    return posEuler
