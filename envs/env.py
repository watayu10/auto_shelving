import socket
import time

import pybullet as p
import pybullet_data
from tqdm import tqdm

from my_utils.create_pybullet_object import create_cuboid, create_onigiri
from my_utils.transform import quat_rotate_vector
from third_party.pybullet_ur5_robotiq.utilities import Camera


# this class is based on
class ProductShelvingEnv:

    SIMULATION_STEP_DELAY = 1 / 240.0

    def __init__(self, robot, vis=False, max_steps=100000):
        self.robot = robot
        self.vis = vis
        if self.vis:
            self.p_bar = tqdm(ncols=0, disable=False)
        self.camera = None  # 環境を上から見下ろすカメラ
        # self.camera2 = Camera()  # 棚を横から見るカメラ

        # define environment
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.planeID = p.loadURDF("plane.urdf")

        self.robot.load()
        self.robot.step_simulation = self.step_simulation

        self.container = create_cuboid(
            [0.20 / 2, 0.30 / 2, 0.01 / 2],
            [59 / 255, 175 / 255, 117 / 255, 1],
            [0.40, 0, 0.005],
            0,
        )
        self.shelf = create_cuboid(
            [0.1 / 2, 0.2 / 2, 0.01 / 2],
            [247 / 255, 246 / 255, 235 / 255, 1],
            [-0.40, 0, 0.25],
            0,
        )

        self.onigiris = create_onigiri()
        self.current_step = 0
        self.max_steps = max_steps

    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        p.stepSimulation()
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)
            self.p_bar.update(1)

    def step(self, action, control_method="joint"):
        """
        action: (x, y, z, roll, pitch, yaw, gripper_opening_length) for End Effector Position Control
                (a1, a2, a3, a4, a5, a6, a7, gripper_opening_length) for Joint Position Control
        control_method:  'end' for end effector position control
                         'joint' for joint position control
        """
        self.current_step += 1
        assert control_method in ("joint", "end")
        self.robot.move_ee(action[:-1], control_method)
        self.robot.move_gripper(action[-1])
        for _ in range(120):  # Wait for a few steps
            self.step_simulation()
        reward = None
        done = False
        info = None
        if self.current_step >= self.max_steps:
            done = True

        return self.get_observation(), reward, done, info

    def get_observation(self):
        obs = dict()
        if isinstance(self.camera, Camera):
            rgb, depth, seg = self.camera.shot()
            obs.update(dict(rgb=rgb, depth=depth, seg=seg))
        else:
            assert self.camera is None
        obs.update(self.robot.get_joint_obs())

        return obs

    def remove_onigiris(self):
        for i in self.onigiris:
            p.removeBody(i)

    def reset(self):
        self.robot.reset()
        self.remove_onigiris()
        self.onigiris = create_onigiri()
        return self.get_observation()

    def close(self):
        p.disconnect(self.physicsClient)

    def draw_ee_arrow(self):
        # ee方向に矢印を表示 eeの座標で[1,0,0]方向 [0,0,-0.1]にカメラを配置
        obs = self.get_observation()
        ee_pos = obs["ee_pos"]
        ori = obs["ee_ori"]
        forward_local = [10, 0, 0]  # 線の長さ
        forward_world = quat_rotate_vector(ori, forward_local)
        local_pos = [0, 0, 0]
        world_pos = quat_rotate_vector(ori, local_pos)
        start_pos = [
            ee_pos[0] + world_pos[0],
            ee_pos[1] + world_pos[1],
            ee_pos[2] + world_pos[2],
        ]
        end_pos = [
            ee_pos[0] + forward_world[0] + world_pos[0],
            ee_pos[1] + forward_world[1] + world_pos[1],
            ee_pos[2] + forward_world[2] + world_pos[2],
        ]
        if self.line_id is not None:
            p.removeUserDebugItem(self.line_id)

        self.line_id = p.addUserDebugLine(
            start_pos,
            end_pos,
            lineColorRGB=[1, 0, 0],
            lineWidth=3,
            lifeTime=0,
        )
