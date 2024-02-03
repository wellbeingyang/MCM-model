import numpy as np
import random
from tools import *


# 是否有动力
propulsion = True


def check_position(pos):
    for i in range(3):
        if pos[i] < 0:
            pos[i] = 0
        elif pos[i] > shape[i]:
            pos[i] = shape[i]
    return pos


# delta_t秒后的情况更新一步
def step_forward(t):
    if propulsion and random.random() < f(t):
        propulsion = False

    if propulsion:
        v_lost = VZ
        pos = update_position(pos, v_lost)
        pos = check_position(pos)
    else:
        v_lost = np.random.normal(loc=v_lost, scale=[sigma]*3, size=3)
        pos = update_position(pos, v_lost)
        F = update_force(pos, v_lost, k, mass, g, density,
                         density_water, current_v, height)
        v_lost = update_speed()
