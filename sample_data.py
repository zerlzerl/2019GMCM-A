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
        sample_rate = 0.99
        sample_num = round(sample_rate * len(file_data))
        sample_data = random.sample(file_data, sample_num)  # 从单个文件中sample的数据
        sampled_data.extend(sample_data)

    index += 1
    # break

# 写入文件保存
random.shuffle(sampled_data)
data_1 = sampled_data[0: 2000000 - 1]
data_2 = sampled_data[2000000: 4000000 - 1]
data_3 = sampled_data[4000000: 6000000 - 1]
data_4 = sampled_data[6000000: 8000000 - 1]
data_5 = sampled_data[8000000: 10000000 - 1]
data_6 = sampled_data[10000000: -1]
sampled_file_path = "./data/sampled_data_1.csv"
with open(sampled_file_path, "w", newline="") as f:
    csv_writer = csv.DictWriter(f, fieldnames=sampled_data[0].keys())
    csv_writer.writeheader()
    for row in data_1:
        csv_writer.writerow(row)

sampled_file_path = "./data/sampled_data_2.csv"
with open(sampled_file_path, "w", newline="") as f:
    csv_writer = csv.DictWriter(f, fieldnames=sampled_data[0].keys())
    csv_writer.writeheader()
    for row in data_2:
        csv_writer.writerow(row)

sampled_file_path = "./data/sampled_data_3.csv"
with open(sampled_file_path, "w", newline="") as f:
    csv_writer = csv.DictWriter(f, fieldnames=sampled_data[0].keys())
    csv_writer.writeheader()
    for row in data_3:
        csv_writer.writerow(row)

sampled_file_path = "./data/sampled_data_4.csv"
with open(sampled_file_path, "w", newline="") as f:
    csv_writer = csv.DictWriter(f, fieldnames=sampled_data[0].keys())
    csv_writer.writeheader()
    for row in data_4:
        csv_writer.writerow(row)

sampled_file_path = "./data/sampled_data_5.csv"
with open(sampled_file_path, "w", newline="") as f:
    csv_writer = csv.DictWriter(f, fieldnames=sampled_data[0].keys())
    csv_writer.writeheader()
    for row in data_5:
        csv_writer.writerow(row)

sampled_file_path = "./data/sampled_data_6.csv"
with open(sampled_file_path, "w", newline="") as f:
    csv_writer = csv.DictWriter(f, fieldnames=sampled_data[0].keys())
    csv_writer.writeheader()
    for row in data_6:
        csv_writer.writerow(row)


