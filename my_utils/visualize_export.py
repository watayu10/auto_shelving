import glob
import os
import pickle
from collections import Counter

import cv2
import numpy as np

folder_path = "./expert/model2_1"
pickle_files = glob.glob(os.path.join(folder_path, "*.pkl"))
loaded_dict = {}
"""
index = None
for file in pickle_files:
    if os.path.basename(file) == "actions.pkl":
        with open(file, "rb") as f:
            data = pickle.load(f)
        action = np.append(
            np.zeros((1, 7)),
            np.diff(np.array(data), axis=0),
            axis=0,
        )
        index = np.where(~np.all(action == 0, axis=1))
        action = action[index]
        loaded_dict["actions.pkl"] = action
    else:
        with open(file, "rb") as f:
            data = pickle.load(f)
        print(type(data))
        temp = np.array(data)[index]
        loaded_dict[os.path.basename(file)] = temp
"""
for file in pickle_files:
    with open(file, "rb") as f:
        loaded_dict[os.path.basename(file)] = pickle.load(f)

for key in loaded_dict.keys():
    print(len(loaded_dict[key]))


# actions = loaded_dict["actions.pkl"]
camera1 = loaded_dict["camera1_images.pkl"]
camera2 = loaded_dict["camera2_images.pkl"]
ee_images = loaded_dict["ee_images.pkl"]


"""
max_value = max(max(row) for row in actions)
min_value = min((min(row) for row in actions))
print(max_value)
print(min_value)

tuple_rows = [tuple(row) for row in actions]
counts = Counter(tuple_rows)

# 表示
for row, count in counts.items():
    print(list(row), "→", count)
"""


def show_images_opencv(images, interval=1000):  # intervalはミリ秒
    for img in images:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # OpenCVはBGRなので変換
        cv2.imshow("Video Playback", img_bgr)
        key = cv2.waitKey(interval)
        if key == 27:  # ESCキーで終了
            break
    cv2.destroyAllWindows()


def show_multiple_image_sequences_with_grid(
    image_lists, interval=100, resize_to=(200, 200), grid_width=10, grid_color=(0, 0, 0)
):
    """
    複数のRGB画像リストを1つのウィンドウで動画再生。
    画像の間にグリッド（罫線）を挿入。

    Parameters:
        - image_lists: [[img1_1, img1_2, ...], [img2_1, img2_2, ...], ...]
        - interval: 再生間隔（ms）
        - resize_to: 画像サイズを統一する (W, H)
        - grid_width: グリッドの太さ（ピクセル単位）
        - grid_color: グリッドの色（BGR形式）
    """
    num_frames = min(len(seq) for seq in image_lists)

    # 縦グリッドを事前に作成（縦に画像が並ぶなら横グリッド）
    if resize_to:
        h, w = resize_to[1], resize_to[0]
    else:
        h, w, _ = image_lists[0][0].shape
    grid = np.full((h, grid_width, 3), grid_color, dtype=np.uint8)  # 縦のグリッド

    for i in range(num_frames):
        frame_imgs = []
        for j, seq in enumerate(image_lists):
            img = seq[i]
            if resize_to:
                img = cv2.resize(img, resize_to)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            frame_imgs.append(img_bgr)
            # グリッドを画像の間に追加（最後の画像の後には追加しない）
            if j < len(image_lists) - 1:
                frame_imgs.append(grid)

        combined = np.hstack(frame_imgs)
        cv2.imshow("Multi Video with Grid", combined)

        key = cv2.waitKey(interval)
        if key == 27:  # ESCキーで終了
            break

    cv2.destroyAllWindows()


# show_multiple_image_sequences_with_grid([camera1, camera2, ee_images])

import cv2
import numpy as np


def save_multiple_image_sequences_with_grid_video(
    image_lists,
    save_path="output.mp4",
    interval=100,
    resize_to=(200, 200),
    grid_width=10,
    grid_color=(0, 0, 0),
    fps=10,
):
    """
    複数のRGB画像リストを横並びでグリッド付きにし、動画として保存。

    Parameters:
        - image_lists: [[img1_1, img1_2, ...], [img2_1, img2_2, ...], ...]
        - save_path: 保存する動画ファイルのパス
        - interval: 表示間隔(ms) → fpsが優先される
        - resize_to: 各画像のサイズを統一（W, H）
        - grid_width: 画像の間に挟むグリッドの幅（ピクセル）
        - grid_color: グリッドの色（BGR）
        - fps: 保存する動画のFPS
    """
    num_frames = min(len(seq) for seq in image_lists)

    # 各画像のサイズを決定
    if resize_to:
        h, w = resize_to[1], resize_to[0]
    else:
        h, w, _ = image_lists[0][0].shape

    # グリッドの準備
    grid = np.full((h, grid_width, 3), grid_color, dtype=np.uint8)

    # 1フレームの横幅計算（画像 + グリッドの幅）
    frame_width = w * len(image_lists) + grid_width * (len(image_lists) - 1)
    frame_height = h

    # 動画ライター初期化
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))

    for i in range(num_frames):
        frame_imgs = []
        for j, seq in enumerate(image_lists):
            img = seq[i]
            if resize_to:
                img = cv2.resize(img, resize_to)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            frame_imgs.append(img_bgr)
            if j < len(image_lists) - 1:
                frame_imgs.append(grid)
        combined = np.hstack(frame_imgs)
        out.write(combined)

    out.release()
    print("保存完了:", save_path)


save_multiple_image_sequences_with_grid_video([camera1, camera2, ee_images])
