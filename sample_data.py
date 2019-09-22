import csv
import os
import random

import numpy as np

import math

file_dir = "E:\\Study\\19华为杯\\赛题\\2019年中国研究生数学建模竞赛A题\\cleaned_train_set"
file_list = os.listdir(file_dir)

sampled_data = []  # 从所有文件中sample的数据
current_f = 2585.0
index = 1
for file_name in file_list:
    print("file: %s, index: %04d" % (file_name, index))
    file_path = os.path.join(file_dir, file_name)
    with open(file_path, "r") as f:
        csv_reader = csv.DictReader(f)

        file_data = []  # 文件完整数据
        for data_line in csv_reader:
            file_data.append(data_line)

        if float(file_data[0]["Frequency Band"]) != current_f:
            current_f = float(file_data[0]["Frequency Band"])
            print("frequency change: %f" % current_f)


        # sample
        sample_rate = 0.1
        sample_num = round(sample_rate * len(file_data))
        sample_data = random.sample(file_data, sample_num)  # 从单个文件中sample的数据
        sampled_data.extend(sample_data)

    index += 1
    # break

# 写入文件保存
random.shuffle(sampled_data)
sampled_file_path = "./data/sampled_data.csv"
with open(sampled_file_path, "w", newline="") as f:
    csv_writer = csv.DictWriter(f, fieldnames=sampled_data[0].keys())
    csv_writer.writeheader()
    for row in sampled_data:
        csv_writer.writerow(row)


