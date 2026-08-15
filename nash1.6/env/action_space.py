import math as m
from typing import List

import numpy as np

# nx 切向过载，ny 法向过载，roll 滚转角，Pitch 限制
# Pitch: 0 = 拉成水平; -1 = 保持当前俯仰角
plane_action_dict_simple = {
    '0': [0.0, 1.0, 0.0, 0],                 # 匀速前飞
    '1': [2.0, 1.0, 0.0, 0],                 # 加速前飞
    '2': [-2.0, 1.0, 0.0, 0],                # 减速前飞
    '3': [0.0, 2.0, 0.0, -1],                # 爬升
    '4': [0.0, 0.0, 0.0, -1],                # 俯冲
    '5': [0.0, 2.0, 0.25 * m.pi, -1],        # 左爬升
    '6': [0.0, 2.0, -0.25 * m.pi, -1],       # 右爬升
    '7': [0.0, -2.0, -0.25 * m.pi, -1],      # 左俯冲
    '8': [0.0, -2.0, 0.25 * m.pi, -1],       # 右俯冲
    '9': [0.0, 2.0, m.acos(1 / 2), 0],       # 左转弯
    '10': [0.0, 2.0, -m.acos(1 / 2), 0],     # 右转弯
}

def get_simple() -> np.ndarray:
    """Return the primitive blue-plane actions in the original ordering."""
    return np.array([value for value in plane_action_dict_simple.values()], dtype=float)


def get_simple_list() -> List[list]:
    """Return primitive actions as a list for easy sequence assembly."""
    return [list(value) for value in plane_action_dict_simple.values()]