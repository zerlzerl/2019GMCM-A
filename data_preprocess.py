import random
import time

import math
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Dense

from data_reader import *


def data_preprocess(data_line):
    # 对一条原始数据行进行处理
    # ========================================
    # 对发射机
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
    rsrp = data_line["RSRP"]  # 接收器信号强度

    # 相对位置
    rel_x = x - cell_x  # 接收器相对于基站的x坐标
    rel_y = y - cell_y  # 接收器相对于基站的y坐标
    rel_h = (cell_h + cell_building_h + cell_altitude) - (building_h + altitude)  # 基站相对于接收器的高度

    # 相对距离
    d = math.sqrt(math.pow(rel_x * 5, 2) + math.pow(rel_y * 5, 2) + math.pow(rel_h, 2))  # 接收器相对于基站的距离，单位是m

    # 下倾角
    downtilt = elec_downtilt + mechan_downtilt
    # 出射方向
    fire_directrion = np.array(
        [math.cos(downtilt) * math.sin(azimuth), math.cos(downtilt) * math.cos(azimuth), -math.sin(downtilt)])
    # 发射机到接收机的向量
    bs2receiver_direction = np.array([rel_x, rel_y, -rel_h])
    # 信号线与实际信号之间的夹角的cos值，考虑分母为零的情况报错
    angle_cos = fire_directrion.dot(bs2receiver_direction) / (np.sqrt(fire_directrion.dot(fire_directrion)) * np.sqrt(
        bs2receiver_direction.dot(bs2receiver_direction)))

    # 将clutter转化为one hot表示形式
    cell_clutter_vec = [0] * 20
    clutter_vec = [0] * 20

    cell_clutter_vec[cell_clutter - 1] = 1
    clutter_vec[clutter - 1] = 1

    # 构建特征向量，[相对距离，相对高度，信号线与实际信号之间的夹角cos，发射功率，发射器地物类型，接收器地物类型]
    train_vec = np.array([d, rel_h, angle_cos, power], dtype=np.float)
    train_vec = np.append(train_vec, cell_clutter_vec)
    train_vec = np.append(train_vec, clutter_vec)
    return train_vec, rsrp


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(44, input_dim=44, kernel_initializer='normal', activation='relu'))
    model.add(Dense(22, kernel_initializer='normal', activation='relu'))
    model.add(Dense(11, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


if __name__ == '__main__':
    training_data = read_training_data()
    test_data = read_test_data()

    preprocessed_training_data = []
    label = []
    for data_line in training_data:
        x, y = data_preprocess(data_line)
        preprocessed_training_data.append(x)
        label.append(y)

    preprocessed_test_data = []
    test_label = []
    for data_line in test_data:
        x, y = data_preprocess(data_line)
        preprocessed_test_data.append(x)
        test_label.append(y)

    train_x = np.array(preprocessed_training_data)
    train_y = np.array(label)

    test_x = np.array(preprocessed_test_data)
    test_y = np.array(test_label)

    # train_mean = train_x.mean(axis=0)
    # train_std = train_x.std(axis=0)
    # train_x = (train_x - train_mean) / train_std
    # test_x = (test_x - train_mean) / train_std

    model = baseline_model()
    # checkpoint = ModelCheckpoint("model/my_model.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='max')
    # callbacks_list = [checkpoint]
    model.fit(train_x, train_y, batch_size=100, validation_split=0.1, epochs=100)

    model.save("model/my_model.h5")
    model.evaluate(test_x, test_y)

    y_pred = model.predict(test_x)

    # 可视化

    test_y_sample = test_y[:100]
    pred_y_sample = y_pred[:100]

    with open("./data/result/result.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "test_rsrp", "pred_rsrp"])
        for i in range(len(test_y_sample)):
            writer.writerow([str(i), str(test_y_sample[i]), str(pred_y_sample[i][0])])