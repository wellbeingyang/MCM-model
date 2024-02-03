import numpy as np
import random
from scipy.stats import norm

# 先只考虑100x100x100的范围
shape = (100, 100, 100)

# 地底高度，用于模拟海底地形
height = np.random.randint(low=70, high=100, size=(shape[0], shape[1]))

# 每个区块内的洋流速度
current_v = np.random.randint(low=0, high=2, size=shape+(3,))

# 每隔delta_t的时间更新一次数据
delta_t = 1

# 搜救潜艇移动速度
v = np.array([4, 4, 4])

# 搜救潜艇当前坐标
r_s = np.array([0, 0, 0])

# 失联潜艇初始位置坐标
r0 = np.array([0, 0, np.random.randint(0, height[0][0])])

# 失联潜艇当前坐标
r = r0

# 雷达探测半径
R = 20

# 概率分布矩阵
P = np.zeros(shape=shape)
P[tuple(r0)] = 1


# 判断是否结束
def finish():
    return False

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


# 原始概率密度函数 P(x, y, z)
def original_probability_density(x, y, z):
    # 示例：一个简单的概率密度函数，你需要根据实际情况替换为你的概率密度函数
    return np.exp(-((x - 5)**2 + (y - 5)**2 + (z - 5)**2) / 10)


# 求解概率密度函数的卷积
def convolve_probability_density(P, v, delta_t):
    x = np.arange(P.shape[0])
    y = np.arange(P.shape[1])
    z = np.arange(P.shape[2])

    # 生成速度的分布函数
    vx, vy, vz = np.meshgrid(norm.pdf(x, scale=v[0]), norm.pdf(
        y, scale=v[1]), norm.pdf(z, scale=v[2]))

    # 卷积操作
    P_convolved = np.fft.ifftn(np.fft.fftn(
        P) * np.fft.fftn(vx) * np.fft.fftn(vy) * np.fft.fftn(vz)).real

    return P_convolved
