import csv
import time

import keras
import math
import tensorflow as tf
import numpy as np


keras.backend.clear_session()
# 原始文件的header顺序
test_file_path = "E:\Code\PyCharm\\2019GMCM_A\data\sampled_data_5w.csv"
test_x = []
test_y = []
with open(test_file_path, "r") as f:
    csv_reader = csv.DictReader(f)
    for data_line in csv_reader:
        cell_idx = int(data_line["Cell Index"])  # Cell ID
        cell_x = float(data_line["Cell X"])  # 单位是栅格
        cell_y = float(data_line["Cell Y"])  # 单位是栅格
        cell_h = float(data_line["Height"])  # 站点高度,单位是m
        cell_building_h = float(data_line["Cell Building Height"])  # 站点所在建筑物高度，单位是m
        cell_altitude = float(data_line["Cell Altitude"])  # 站点海拔，单位是m
        cell_clutter = int(data_line["Cell Clutter Index"])  # 站点所在地物类型索引

        azimuth = float(data_line["Azimuth"])  # 方向角，单位是°
        elec_downtilt = float(data_line["Electrical Downtilt"])  # 电下倾角，单位是°
        mechan_downtilt = float(data_line["Mechanical Downtilt"])  # 机械下倾角，单位是°
        frequency = float(data_line["Frequency Band"])  # 带宽，单位MHz
        power = float(data_line["RS Power"])  # 发射功率，单位dBm

        # ========================================
        # 对接收器
        x = float(data_line["X"])  # 接收器x坐标，单位是栅格
        y = float(data_line["Y"])  # 接收器y坐标，单位是栅格
        building_h = float(data_line["Building Height"])  # 接收器所在地建筑物高度，单位是m
        altitude = float(data_line["Altitude"])  # 接收器所在地海拔高度，单位是m
        clutter = int(data_line["Clutter Index"])  # 接收器所在地物类型索引

        # 标签
        rsrp = float(data_line["RSRP"])  # 接收器信号强度

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
        # 分七个类
        classes = {
            "10": 0, "11": 0, "12": 0, "20": 0, "13": 1, "14": 1, "16": 1, "15": 2, "18": 2,
            "17": 3, "19": 3, "1": 4, "2": 4, "3": 4, "4": 5, "5": 5, "6": 5, "7": 6, "8": 6, "9": 6
        }
        cell_clutter_vec = [0] * 7  # 发射机所在地物特征
        clutter_vec = [0] * 7  # 接收机所在地物特征

        cell_clutter_vec[classes[str(cell_clutter)]] = 1
        clutter_vec[classes[str(clutter)]] = 1

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
                round(cell_idx / 100)  # 小区去掉最后两位
            ]
        feature_list.extend(cell_clutter_vec)
        feature_list.extend(clutter_vec)
        features = np.asarray(feature_list, dtype=np.float32)
        if np.any(np.isnan(features)) or np.any(np.isnan(rsrp)):
            print("cell_idx: %f, x: %f, y: %f" % (cell_idx, x, y))
            continue
        else:
            test_x.append(features)
            test_y.append(rsrp)

test_x = np.array(test_x)
test_y = np.array(test_y)
# 测h5模型对不对
h5_model_path = "./model/0922/weights-improvement-136-58.89.h5"
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

