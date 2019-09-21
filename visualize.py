# 从csv读取原始数据
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

dir_path = 'E:\\Study\\19华为杯\赛题\\2019年中国研究生数学建模竞赛A题\\train_set'
output_dir = './vis/'

for file in os.listdir(dir_path):
    print('processing: ' + file)
    x, y, r = [], [], []
    file_path = os.path.join(dir_path, file)
    if not os.path.isdir(file_path):
        print(file_path)
        with open(file_path, "r") as f:
            file_reader = csv.DictReader(f)
            for row in file_reader:
                x.append(int(row["X"]))
                y.append(int(row["Y"]))
                r.append(float(row["RSRP"]))
                x0 = round(float(row["Cell X"]))
                y0 = round(float(row["Cell Y"]))
    x, y, r = np.array(x), np.array(y), np.array(r)

    # xx, yy = np.meshgrid(np.linspace(min(x), max(x)), np.linspace(min(y), max(y)))
    # z =

    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    # plt.bar(x, y, r, marker='.', alpha=0.8)
    ax.scatter([x0], [y0], [-75.0], c='r', s=100, marker='o')
    ax.scatter(x, y, r, c='b', marker='^', s=1, alpha=.3)
    # ax.bar(x, y, r)
    # ax.plot_surface(x,y,r,rstride=2,cstride=1,cmap=plt.cm.coolwarm,alpha=0.8)  #绘制三维图表面

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter([x0], [y0], c='r', s=100, marker='o')
    ax2.scatter(x, y, c='b', marker='^', s=1, alpha=.3)

    plt.savefig(output_dir + file.split('.')[0] + '.png', dpi=400, bbox_inches='tight')

    # plt.show()
    # break
