import math
import os
import sys

import numpy as np
import pybullet as p
import pybullet_data
from tqdm import tqdm

sys.path.append(os.path.abspath("third_party/pybullet_ur5_robotiq"))

from env import ClutteredPushGrasp
from robot import RobotBase
from utilities import Models


# Wrapper to resolve relative path issues in pybullet_ur5_robotiq
class ClutteredPushGraspWrapper(ClutteredPushGrasp):
    def __init__(self, robot, models: Models, camera=None, vis=False) -> None:
        self.robot = robot
        self.vis = vis
        if self.vis:
            self.p_bar = tqdm(ncols=0, disable=False)
        self.camera = camera

        # define environment
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.planeID = p.loadURDF("plane.urdf")

        self.robot.load()
        self.robot.step_simulation = self.step_simulation

        # custom sliders to tune parameters (name of the parameter,range,initial value)
        self.xin = p.addUserDebugParameter("x", -0.224, 0.224, 0)
        self.yin = p.addUserDebugParameter("y", -0.224, 0.224, 0)
        self.zin = p.addUserDebugParameter("z", 0, 1.0, 0.5)
        self.rollId = p.addUserDebugParameter("roll", -3.14, 3.14, 0)
        self.pitchId = p.addUserDebugParameter("pitch", -3.14, 3.14, np.pi / 2)
        self.yawId = p.addUserDebugParameter("yaw", -np.pi / 2, np.pi / 2, np.pi / 2)
        self.gripper_opening_length_control = p.addUserDebugParameter(
            "gripper_opening_length", 0, 0.085, 0.04
        )
        # just change below path
        self.boxID = p.loadURDF(
            "third_party/pybullet_ur5_robotiq/urdf/skew-box-button.urdf",
            [0.0, 0.0, 0.0],
            # p.getQuaternionFromEuler([0, 1.5706453, 0]),
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
            flags=p.URDF_MERGE_FIXED_LINKS | p.URDF_USE_SELF_COLLISION,
        )

        # For calculating the reward
        self.box_opened = False
        self.btn_pressed = False
        self.box_closed = False


# add ee_ori
class RobotBaseWrapper(RobotBase):
    def get_joint_obs(self):
        positions = []
        velocities = []
        for joint_id in self.controllable_joints:
            pos, vel, _, _ = p.getJointState(self.id, joint_id)
            positions.append(pos)
            velocities.append(vel)
        ee_pos = p.getLinkState(self.id, self.eef_id)[0]
        ee_ori = p.getLinkState(self.id, self.eef_id)[1]
        return dict(
            positions=positions, velocities=velocities, ee_pos=ee_pos, ee_ori=ee_ori
        )


# change RobotBase to RobotBaseWrapper
class UR5Robotiq85(RobotBaseWrapper):
    def __init_robot__(self):
        self.eef_id = 7
        self.arm_num_dofs = 6
        self.arm_rest_poses = [
            -1.5690622952052096,
            -1.5446774605904932,
            1.343946009733127,
            -1.3708613585093699,
            -1.5707970583733368,
            0.0009377758247187636,
        ]
        self.id = p.loadURDF(
            "third_party/pybullet_ur5_robotiq/urdf/ur5_robotiq_85.urdf",
            self.base_pos,
            self.base_ori,
            useFixedBase=True,
            flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES,
        )
        self.gripper_range = [0, 0.085]

    def __post_load__(self):
        # To control the gripper
        mimic_parent_name = "finger_joint"
        mimic_children_names = {
            "right_outer_knuckle_joint": 1,
            "left_inner_knuckle_joint": 1,
            "right_inner_knuckle_joint": 1,
            "left_inner_finger_joint": -1,
            "right_inner_finger_joint": -1,
        }
        self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)

    def __setup_mimic_joints__(self, mimic_parent_name, mimic_children_names):
        self.mimic_parent_id = [
            joint.id for joint in self.joints if joint.name == mimic_parent_name
        ][0]
        self.mimic_child_multiplier = {
            joint.id: mimic_children_names[joint.name]
            for joint in self.joints
            if joint.name in mimic_children_names
        }

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(
                self.id,
                self.mimic_parent_id,
                self.id,
                joint_id,
                jointType=p.JOINT_GEAR,
                jointAxis=[0, 1, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0],
            )
            p.changeConstraint(
                c, gearRatio=-multiplier, maxForce=100, erp=1
            )  # Note: the mysterious `erp` is of EXTREME importance

    def move_gripper(self, open_length):
        # open_length = np.clip(open_length, *self.gripper_range)
        open_angle = 0.715 - math.asin(
            (open_length - 0.010) / 0.1143
        )  # angle calculation
        # Control the mimic gripper joint(s)
        p.setJointMotorControl2(
            self.id,
            self.mimic_parent_id,
            p.POSITION_CONTROL,
            targetPosition=open_angle,
            force=self.joints[self.mimic_parent_id].maxForce,
            maxVelocity=self.joints[self.mimic_parent_id].maxVelocity,
        )
