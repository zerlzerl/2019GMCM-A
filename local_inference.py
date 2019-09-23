import csv
import time

import math
import tensorflow as tf
from tensorflow.python import keras
import numpy as np


keras.backend.clear_session()
# 原始文件的header顺序
test_file_path = "../MathModel/train_set/train_130901.csv"
test_x = []
test_y = []
with open(test_file_path, "r") as f:
    csv_reader = csv.reader(f)
    next(f)
    for data_line in csv_reader:
        one_line_data = np.array(data_line, np.float32)
        cell_x = one_line_data[1]
        cell_y = one_line_data[2]
        cell_h = one_line_data[3]
        azimuth = one_line_data[4]
        elec_downtilt = one_line_data[5]
        mechan_downtilt = one_line_data[6]
        frequency = one_line_data[7]
        power = one_line_data[8]
        cell_altitude = one_line_data[9]
        cell_building_h = one_line_data[10]
        cell_clutter = int(one_line_data[11])
        x = one_line_data[12]
        y = one_line_data[13]
        altitude = one_line_data[14]
        building_h = one_line_data[15]
        clutter = int(one_line_data[16])
        rsrp = one_line_data[17]

        # 电气特征
        h_s = cell_h  # 发射机相对地面有效高度
        horizon_angel = azimuth  # 水平方向角
        downtilt = elec_downtilt + mechan_downtilt  # 下倾角

        # 几何特征
        d = math.sqrt((x - cell_x) ** 2 + (y - cell_y) ** 2)  # 水平面上的链路距离

        fire_directrion = np.array(
            [math.cos(downtilt) * math.sin(horizon_angel), math.cos(downtilt) * math.cos(horizon_angel),
             -math.sin(downtilt)])
        bs2receiver_direction = np.array([x - cell_x, y - cell_y, (cell_h + cell_altitude) - altitude])
        # 信号线与实际信号之间的夹角的cos值，考虑分母为零的情况报错
        beta_cos = fire_directrion.dot(bs2receiver_direction) / (
                np.sqrt(fire_directrion.dot(fire_directrion)) * np.sqrt(
            bs2receiver_direction.dot(bs2receiver_direction)))

        # 接收点到信号线的垂直距离
        h_v = math.sqrt(
            (x - cell_x) ** 2 + (y - cell_y) ** 2 + (cell_h + cell_altitude - altitude) ** 2) * math.sqrt(
            1 - beta_cos ** 2)

        # 环境特征
        cell_clutter_vec = [0] * 20  # 发射机所在地物特征
        clutter_vec = [0] * 20  # 接收机所在地物特征
        cell_clutter_vec[cell_clutter - 1] = 1
        clutter_vec[clutter - 1] = 1

        h_build_s = cell_building_h
        h_build_r = building_h
        alti_diff = cell_altitude - altitude

        # 不可训练特征
        p_t = power
        f = frequency

        # 构造feature list
        feature_list = \
            [
                # 电器特征
                power,  # 发射功率
                math.log(abs(h_s + 1)),
                horizon_angel,
                downtilt,
                math.log(f),  # 先放进去试试
                # 几何特征
                math.log(d / 1000 + 1),
                beta_cos,
                math.log(h_v / 1000 + 1),
                math.log(d / 1000 + 1) * math.log(abs(h_s + 1)),
                # 环境特征
                h_build_s,
                h_build_r,
                alti_diff,
            ]
        feature_list.extend(cell_clutter_vec)
        feature_list.extend(clutter_vec)

        features = np.asarray(feature_list, dtype=np.float32)
        test_x.append(features)
        test_y.append(rsrp)

test_x = np.array(test_x)
test_y = np.array(test_y)
# 测h5模型对不对
h5_model_path = "./model/my_model.h5"
tf.keras.backend.set_learning_phase(0)
model = keras.models.load_model(h5_model_path)

pred_y = model.predict(test_x)
# test_y = test_y.reshape(-1, 1)
pred_y = pred_y.reshape(-1,)

mean_pred_y = pred_y.mean()
mean_test_y = test_y.mean()
print("mean_pred_y:" + str(mean_pred_y))
print("mean_test_y:" + str(mean_test_y))

std_pred_y = pred_y.std()
std_test_y = test_y.std()
print("std_pred_y:" + str(std_pred_y))
print("std_test_y:" + str(std_test_y))

tp = 0
fp = 0
fn = 0
tn = 0
for i in range(len(test_y)):
    if test_y[i] < -103 and pred_y[i] < -103:
        tp += 1
    if test_y[i] < -103 and pred_y[i] > -103:
        fn += 1
    if test_y[i] > -103 and pred_y[i] < -103:
        fp += 1
    if test_y[i] > -103 and pred_y[i] > -103:
        tn += 1

precision = tp / (tp + fp)
print("precision:" + str(precision))
recall = tp / (tp + fn)
print("recall:" + str(recall))
pcrr = 2 * precision * recall / (precision + recall)
print("PCRR:" + str(pcrr))

msre = np.sqrt(((np.array(test_y) - np.array(pred_y)) ** 2).mean())
print("MSRE:" + str(msre))

