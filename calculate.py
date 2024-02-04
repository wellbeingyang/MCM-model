import numpy as np
import random
from tools import *
import tools


# delta_t秒后的情况更新一步
def step_forward(t):
    if tools.propulsion and random.random() < p_disable(t):
        tools.propulsion = False

    if tools.propulsion:
        tools.v_lost = [0, 0, tools.VZ]
        tools.pos = update_position(tools.pos, tools.v_lost)
    else:
        tools.v_lost = np.random.normal(
            loc=tools.v_lost, scale=[tools.sigma]*3, size=3)
        tools.pos = update_position(tools.pos, tools.v_lost)
        tools.F = update_force(tools.pos, tools.v_lost, tools.k, tools.mass, tools.g, tools.density,
                                   tools.density_water, tools.current_v, tools.height)
        tools.v_lost = update_speed(
            tools.pos, tools.v_lost, tools.height)
