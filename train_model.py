import csv
import numpy as np
import math

# data set
from tensorflow.python.keras import backend
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.optimizers import Adam

file_path = "./data/sampled_data_50w.csv"

# total_data = []
# total_label = []
total_x_p = []  # 非弱覆盖
total_y_p = []
total_x_n = []  # 弱覆盖
total_y_n = []
with open(file_path, "r") as f:
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

        # ========================================
        # 候选特征
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

        if np.any(np.isnan(features)) or np.any(np.isnan(rsrp)):
            print("cell_idx: %f, x: %f, y: %f" % (cell_idx, x, y))
            continue
        else:
            if rsrp > -103:
                total_x_p.append(features)
                total_y_p.append(rsrp)
            else:
                total_x_n.append(features)
                total_y_n.append(rsrp)

# 减少正例样本
print("非弱覆盖数量： %d" % len(total_x_p))
print("弱覆盖数量： %d" % len(total_x_n))
sample_pos_rate = 0.5
sample_neg_rate = 1

sample_x_p = total_x_p[:round(sample_pos_rate * len(total_x_p)) - 1]
sample_y_p = total_y_p[:round(sample_pos_rate * len(total_x_p)) - 1]

sample_x_n = total_x_n[:round(sample_neg_rate * len(total_x_n)) - 1]
sample_y_n = total_y_n[:round(sample_neg_rate * len(total_x_n)) - 1]
total_x = np.vstack((np.asarray(sample_x_p), np.asarray(sample_x_n)))
total_y = np.concatenate((np.array(sample_y_p), np.array(sample_y_n))).reshape(-1, 1)
total = np.concatenate((total_x, total_y), axis=1)
np.random.shuffle(total)
total_x = total[:, 0: total.shape[1] - 1]
total_y = total[:, -1]

# 建立模型

# create model
model = Sequential()
model.add(Dense(128, input_dim=total_x.shape[1], kernel_initializer='normal', activation='relu'))
model.add(Dense(64, kernel_initializer='normal', activation='relu'))
model.add(Dense(32, kernel_initializer='normal', activation='relu'))
model.add(Dense(16, kernel_initializer='normal', activation='relu'))
model.add(Dropout())
model.add(Dense(1, kernel_initializer='normal'))
# Compile model
opt = Adam(lr=0.002, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=opt, loss='mse', metrics=['mae'])

# 设置模型保存位置
filepath = "./model/0922/weights-improvement-{epoch:02d}-{loss:.2f}.h5"
checkpoint = ModelCheckpoint(filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='auto')

# 当val_loss不再提升，则停止训练
earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=10,
                          verbose=1,
                          mode='auto')

callbacks = [checkpoint, earlystop]

model.fit(total_x, total_y, validation_split=0.15, batch_size=100, epochs=100, callbacks=callbacks, verbose=1)

model.save("model/my_model.h5")

# 直接测试
# 原始文件的header顺序
test_file_path = "E:\\Study\\19华为杯\\赛题\\2019年中国研究生数学建模竞赛A题\\train_set\\train_2915501.csv"
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

pred_y = model.predict(test_x)
print(pred_y)

