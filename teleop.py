import pickle
import threading
import time

import cv2
import keyboard

from envs.env import ProductShelvingEnv
from my_utils.transform import getPosEulerFromObs, quat_rotate_vector
from my_wrappers.pybullet_ur5_robotiq_wrapper import UR5Robotiq85
from third_party.pybullet_ur5_robotiq.utilities import Camera

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
latest_action = init_action
action_lock = threading.Lock()

# if you want to save data , turn True
data_collection = True

ee_images = []
camera1_images = []
camera2_images = []
actions_list = []


def teleop_keyborad():
    global latest_action
    step = 0.01  # 変化量（小さいほど繊細）
    print("Keyboard control started. Use WASDQE to move, R/F to change z, ESC to exit.")

    while True:
        actions = [0, 0, 0, 0, 0, 0, 0]
        # 終了キー
        if keyboard.is_pressed("esc"):
            break
        # キーによる位置制御
        if keyboard.is_pressed("w"):
            actions[1] += step
        if keyboard.is_pressed("s"):
            actions[1] -= step
        if keyboard.is_pressed("a"):
            actions[0] -= step
        if keyboard.is_pressed("d"):
            actions[0] += step
        if keyboard.is_pressed("up"):
            actions[2] += step
        if keyboard.is_pressed("down"):
            actions[2] -= step
        if keyboard.is_pressed("o"):
            actions[6] += step
        if keyboard.is_pressed("l"):
            actions[6] -= step

        for i in range(len(actions)):
            init_action[i] += actions[i]

        with action_lock:
            latest_action = init_action
        time.sleep(0.05)


def env_loop():
    camera1 = Camera((0, -1, 1.25), (0, 0, 0), (0, 0, 1), 0.1, 5, (224, 224), 40)
    camera2 = Camera((0, 0, 0.5), (-0.40, 0, 0.25), (0, 0, 1), 0.1, 5, (224, 224), 40)
    robot = UR5Robotiq85((0, 0.5, 0), (0, 0, 0))
    env = ProductShelvingEnv(robot, vis=True)
    env.reset()
    # env.SIMULATION_STEP_DELAY = 0
    while True:
        with action_lock:
            # gripper_open_lenthの調整0から0.1の範囲に制限
            open_length = max(0, min(0.1, latest_action[-1]))
            latest_action[-1] = open_length
            obs, reward, done, info = env.step(latest_action, "end")
            if data_collection:
                actions_list.append(latest_action.copy())

        # print(obs, reward, done, info)

        # Mount a camera at the tip of the end effector
        # camera pos is current ee_pos + ee_local pos[0,0,-0.1]
        ee_pos = obs["ee_pos"]
        ori = obs["ee_ori"]
        camera_local_pos = [0, 0, -0.05]
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
        ee_image, _, _ = camera_ee.shot()
        camera1_image, _, _ = camera1.shot()
        camera2_image, _, _ = camera2.shot()

        bgr_ee_image = cv2.cvtColor(ee_image, cv2.COLOR_RGB2BGR)
        bgr_camera1_image = cv2.cvtColor(camera1_image, cv2.COLOR_RGB2BGR)
        bgr_camera2_image = cv2.cvtColor(camera2_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("ee", bgr_ee_image)
        cv2.imshow("1", bgr_camera1_image)
        cv2.imshow("2", bgr_camera2_image)
        cv2.waitKey(1)

        if data_collection:
            ee_images.append(ee_image)
            camera1_images.append(camera1_image)
            camera2_images.append(camera2_image)
        if keyboard.is_pressed("esc"):
            with open("actions.pkl", "wb") as f:
                pickle.dump(actions_list, f)
            with open("ee_images.pkl", "wb") as f:
                pickle.dump(ee_images, f)
            with open("camera1_images.pkl", "wb") as f:
                pickle.dump(camera1_images, f)
            with open("camera2_images.pkl", "wb") as f:
                pickle.dump(camera2_images, f)
            env.close()
            break


keyboard_thread = threading.Thread(target=teleop_keyborad)
keyboard_thread.start()

env_thread = threading.Thread(target=env_loop)
env_thread.start()
