import random

import pybullet as p


def create_cuboid(size, rgba, pos, mass, euler=None, dynamics=None):
    """
    size:[x,y,z]
    rgba:[r,g,b,a]
    mass:
    euler:[x,y,z]
    dynamics:[lateralFriction,spinningFriction,rollingFriction]
    """
    ori = None
    if euler is not None:
        ori = p.getQuaternionFromEuler(euler)

    half_extents = [size[0], size[1], size[2]]
    # 衝突形状を作成
    box_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    # 見た目（オプション）
    box_visual_shape = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=[rgba[0], rgba[1], rgba[2], rgba[3]],
    )
    # ボックスを作成（位置、高さ1mのところに置く）
    box_body = p.createMultiBody(
        baseMass=mass,  # 0にすると固定される
        baseCollisionShapeIndex=box_collision_shape,
        baseVisualShapeIndex=box_visual_shape,
        basePosition=[pos[0], pos[1], pos[2]],
        baseOrientation=ori,
    )
    if dynamics is not None:
        p.changeDynamics(
            box_body,
            -1,
            lateralFriction=dynamics[0],
            spinningFriction=dynamics[1],
            rollingFriction=dynamics[2],
        )

    return box_body


def create_onigiri(num=4):
    size = [0.03 / 4, 0.095 / 4, 0.08 / 4]

    rgbas = [
        [1, 1, 1, 1],  # 白色
        [115 / 255, 78 / 255, 48 / 255, 1],  # 茶色
        [102 / 225, 136 / 225, 0, 1],  # 緑
        [0, 0, 1, 1],  # 青
    ]

    poss = [[0.325 + 0.05 * i, -0.15 / 2, 0.01 + 0.04] for i in range(4)] + [
        [0.325 + 0.05 * i, 0.15 / 2, 0.01 + 0.04] for i in range(4)
    ]
    """poss = (
        [[0.325 + 0.05 * i, -0.15 / 2, 0.01 + 0.04] for i in range(4)]
        + [[0.35 + 0.05 * i, 0, 0.01 + 0.04] for i in range(3)]
        + [[0.325 + 0.05 * i, 0.15 / 2, 0.01 + 0.04] for i in range(4)]
    )

    rgbas = [
        [1, 0, 0, 1],  # red
        [1, 165 / 225, 0, 1],  # orange
        [1, 1, 0, 1],  # yellow
        [0, 128 / 255, 0, 1],  # green
        [0, 1, 1, 1],  # aqua
        [0, 0, 1, 1],  # blue
        [128 / 255, 0, 128 / 255, 1],  # purple
    ]"""

    mass = 0.05
    # onigiri_num = random.randint(1, num)
    # onigiri_colors = random.choices(rgbas, k=onigiri_num)
    # onigiri_poss = random.sample(poss, onigiri_num)
    onigiri_ids = []
    for i in range(1):
        onigiri_ids.append(create_cuboid(size, rgbas[1], poss[3], mass))

    return onigiri_ids
