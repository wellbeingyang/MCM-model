import numpy as np
from scipy.stats import norm

# 先只考虑100x100x100的范围
shape = (100, 100, 100)

# 地底高度，用于模拟海底地形
height = np.random.randint(low=70, high=100, size=(shape[0], shape[1]))

# 每个区块内的洋流速度
current_v = np.random.randint(low=0, high=2, size=shape+(3,))

# 每个区块内的海水密度
density_water = np.random.uniform(low=0.5, high=1, size=shape)

# 每隔delta_t的时间更新一次数据
delta_t = 1

# 用于正态分布的标准差
sigma = 1

# 搜救潜艇移动速度
v_s = np.array([4, 4, 4])

# 搜救潜艇的位置坐标
pos_s = np.array([0, 0, 0])

# 失联潜艇质量(用于从受力计算加速度)
mass = 100

# 失联潜艇密度(用密度、质量、重力加速度及海水密度就可算出浮力)
density = 0.8

# 重力加速度
g = 9.8

# 阻力系数
k = 1

# 约定不失去动力时向上的速度
VZ = -10

# 失联潜艇在某秒计算开始时的位置、速度、受力
pos = np.array([50, 50, np.random.randint(0, height[0][0])])

v_lost = np.array([10, 10, 10])  # 后续修改“初值”为失联时报告的速度

F = np.array([0, 0, 0])

# 失联潜艇在某秒开始时加权前的猜测速度
v_before=np.array([10, 10, 10])

# 雷达探测半径
R = 20

# 概率分布矩阵
P = np.zeros(shape=shape)
P[tuple(pos)] = 1

# 是否有动力
propulsion = True


# 确定t时间失去动力概率的函数
def p_disable(t):
    return 0.5


# 判断是否结束
def finish():
    return ((pos-pos_s)**2).sum() <= R**2


# 创建指示函数 I_R(x, y, z)
def indicator_function(x, y, z):
    return x**2+y**2+z**2 < R**2


# 更新概率分布
def update_probability_distribution(P, indicator_func):
    P_updated = P * np.array([[[(1 - indicator_func(x, y, z)) for z in range(shape[2])]
                             for y in range(shape[1])] for x in range(shape[0])])
    normalization_factor = np.sum(P_updated)
    P_updated /= normalization_factor
    return P_updated


# 求解概率密度函数的卷积
def convolve_probability_density(P, v):
    x = np.arange(P.shape[0])
    y = np.arange(P.shape[1])
    z = np.arange(P.shape[2])

    # 生成速度的分布函数
    vx, vy, vz = np.meshgrid(norm.pdf(x, loc=v[0], scale=sigma), norm.pdf(
        y, loc=v[1], scale=sigma), norm.pdf(z, loc=v[2], scale=sigma))

    # 卷积操作
    P_convolved = np.fft.ifftn(np.fft.fftn(
        P) * np.fft.fftn(vx) * np.fft.fftn(vy) * np.fft.fftn(vz)).real

    return P_convolved


# 更新失联潜艇位置
def update_position(pos, v_lost):
    pos_new_x = pos[0]+v_lost[0]*delta_t
    pos_new_y = pos[1]+v_lost[1]*delta_t
    pos_new_z = pos[2]+v_lost[2]*delta_t
    return np.array([pos_new_x, pos_new_y, pos_new_z])


# 判断是否触底来更新失联潜艇速度
def update_speed(pos, v_lost, height):
    if pos[2] >= height[int(pos[0]), int(pos[1])]:  # 触底：速度归零并调整z方向位置
        pos[2] = height[int(pos[0]), int(pos[1])]
        return np.array([0, 0, 0])
    else:  # 未触底：正常更新
        v_lost_new_x = v_lost[0]+F[0]/mass*delta_t
        v_lost_new_y = v_lost[1]+F[1]/mass*delta_t
        v_lost_new_z = v_lost[2]+F[2]/mass*delta_t
        return np.array([v_lost_new_x, v_lost_new_y, v_lost_new_z])
def update_speed_prediction(pos, v_lost, height, force):
    if pos[2] >= height[int(pos[0]), int(pos[1])]:  # 触底：速度归零并调整z方向位置
        pos[2] = height[int(pos[0]), int(pos[1])]
        return np.array([0, 0, 0])
    else:  # 未触底：正常更新
        v_lost_new_x = v_lost[0]+force[0]/mass*delta_t
        v_lost_new_y = v_lost[1]+force[1]/mass*delta_t
        v_lost_new_z = v_lost[2]+force[2]/mass*delta_t
        return np.array([v_lost_new_x, v_lost_new_y, v_lost_new_z])


# 在更新失联潜艇位置后更新失联潜艇受力情况
def update_force(pos, v_lost, k, mass, g, density, density_water, current_v, height):
    # 更新x、y方向受力，使用所在位置处的相对速度
    F_new_x = -k*(v_lost[0]-current_v[int(pos[0]),
                  int(pos[1]), int(pos[2]), 0])
    F_new_y = -k*(v_lost[1]-current_v[int(pos[0]),
                  int(pos[1]), int(pos[2]), 1])

    # 更新z方向受力，除阻力一项外还有浮力与重力的差
    # 与height统一，定义z方向向下为正
    z_force_f = -k*(v_lost[2]-current_v[int(pos[0]),
                    int(pos[1]), int(pos[2]), 2])
    z_force_G = mass*g
    z_force_Float = density_water[int(pos[0]), int(
        pos[1]), int(pos[2])]*g*(mass/density)
    F_new_z = z_force_f+z_force_G-z_force_Float

    # 如果已经触底且z方向合力向下，置为0
    if pos[2] == height[int(pos[0]), int(pos[1])] and F_new_z > 0:
        F_new_z = 0
    return np.array([F_new_x, F_new_y, F_new_z])
def update_force_prediction(pos, v_lost, k, mass, g, density, density_water, current_v, height):
    # 更新x、y方向受力，使用所在位置处的相对速度
    F_new_x = -k*(v_lost[0]-current_v[0])
    F_new_y = -k*(v_lost[1]-current_v[1])

    # 更新z方向受力，除阻力一项外还有浮力与重力的差
    # 与height统一，定义z方向向下为正
    z_force_f = -k*(v_lost[2]-current_v[2])
    z_force_G = mass*g
    z_force_Float = density_water*g*(mass/density)
    F_new_z = z_force_f+z_force_G-z_force_Float

    # 如果已经触底且z方向合力向下，置为0
    if pos[2] == height[int(pos[0]), int(pos[1])] and F_new_z > 0:
        F_new_z = 0
    return np.array([F_new_x, F_new_y, F_new_z])

def check_position(pos):
    for i in range(3):
        if pos[i] < 0:
            pos[i] = 0
        elif pos[i] >shape[i]:
            pos[i] = shape[i]
    return pos