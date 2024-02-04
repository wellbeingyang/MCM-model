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


t = 0


def save_heat_map():
    p_xy = np.sum(tools.P, axis=2)
    p_xy /= p_xy.sum()
    p_xz = np.sum(tools.P, axis=1)
    p_xz /= p_xz.sum()

    plt.clf()
    plt.imshow(p_xy.T, cmap='viridis', interpolation='nearest', origin='lower')
    plt.colorbar()
    plt.scatter(tools.pos[0], tools.pos[1], color="red",
                marker="o", label="Real position")
    plt.scatter(tools.pos_s[0], tools.pos_s[1], color="orange",
                marker="o", label="Searching submarine")
    plt.title("Heatmap of Probability Distribution")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.savefig(f"results/{dir}/img/{t}-xy.png")

    plt.clf()
    plt.imshow(p_xz.T, cmap='viridis', interpolation='nearest', origin='lower')
    plt.colorbar()
    plt.scatter(tools.pos[0], tools.pos[2], color="red",
                marker="o", label="Real position")
    plt.scatter(tools.pos_s[0], tools.pos_s[2], color="orange",
                marker="o", label="Searching submarine")
    plt.title("Heatmap of Probability Distribution")
    plt.xlabel("X-axis")
    plt.ylabel("Z-axis")
    plt.legend()
    plt.savefig(f"results/{dir}/img/{t}-xz.png")


while (not tools.finished):
    print(f"Current search time: {t} seconds.")
    f = open(file=filename, mode="a", encoding="utf-8")
    f.write(f"## {t}s\n")
    f.write("Current position of the lost submersible:\n")
    f.write(f"x: {tools.pos[0]} y: {tools.pos[1]} z: {tools.pos[2]}\n")
    f.write("Current velocity of the lost submersible:\n")
    f.write(f"{tools.v_lost}\n")
    f.write(f"Predicted probability distribution:\n")
    f.write(f'<figure class="half">\n\t<img src="img/{t}-xy.png">\n\t<img src="img/{t}-xz.png">\n</figure>\n')
    f.write("Current position of the searching submarine:\n")
    f.write(f"x: {tools.pos_s[0]} y: {tools.pos_s[1]} z: {tools.pos_s[2]}\n\n")
    save_heat_map()
    predict.step_forward(t)
    calculate.step_forward(t)
    t += tools.delta_t
    f.close()
