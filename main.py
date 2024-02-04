import numpy as np
import matplotlib.pyplot as plt
import time
import os
from tools import *
import tools
import predict
import calculate

if not os.path.exists("results"):
    os.mkdir("results")

dir = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))

filename = f"results/{dir}/results.md"

os.mkdir(f"results/{dir}")
os.mkdir(f"results/{dir}/img")

f = open(file=filename, mode="w", encoding="utf-8")

t = 0


def save_heat_map():
    p_xy = np.sum(tools.P, axis=2)
    p_xy /= p_xy.sum()

    plt.clf()

    # 绘制热力图
    plt.imshow(p_xy, cmap='viridis', interpolation='nearest', origin='lower')

    # 添加颜色条
    plt.colorbar()

    plt.scatter(tools.pos[0], tools.pos[1], color="red", marker="o", label="Real position")
    plt.scatter(tools.pos_s[0], tools.pos_s[1], color="orange",
                marker="o", label="Searching submarine")

    # 添加标题和轴标签
    plt.title("Heatmap of Probability Distribution")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()

    # 显示图形
    plt.savefig(f"results/{dir}/img/{t}.png")


while (not tools.finished):
    print(f"Current search time: {t} seconds.")
    f.write(f"## {t}s\n")
    f.write("Current position of the lost submersible:\n")
    f.write(f"x: {tools.pos[0]} y: {tools.pos[1]} z: {tools.pos[2]}\n")
    f.write(f"Predicted probability distribution:\n")
    f.write(f'![{t}.png](img/{t}.png "{t}.png")\n')
    f.write("Current position of the searching submarine:\n")
    f.write(f"x: {tools.pos_s[0]} y: {tools.pos_s[1]} z: {tools.pos_s[2]}\n\n")
    save_heat_map()
    predict.step_forward(t)
    calculate.step_forward(t)
    t += 1

f.close()
