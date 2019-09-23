import csv
import random

import numpy as np
import math

# data set
import tensorflow.contrib.learn as skflow
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import *
from sklearn.model_selection import train_test_split
import matplotlib as plt

file_path = "./data/sampled_data_5w.csv"

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
            if rsrp > -103:
                total_x_p.append(features)
                total_y_p.append(rsrp)
            else:
                total_x_n.append(features)
                total_y_n.append(rsrp)

# 减少正例样本
print("非弱覆盖数量： %d" % len(total_x_p))
print("弱覆盖数量： %d" % len(total_x_n))
sample_pos_rate = 0.8
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

X_train, X_test, y_train, y_test = train_test_split(total_x, total_y, test_size=0.1)



# # Network Design
# # --------------
# feature_columns = [tf.feature_column.numeric_column('myInput', shape=(27,))]
#
# STEPS_PER_EPOCH = 100
# EPOCHS = 1000
# BATCH_SIZE = 100
#
# hidden_layers = [128, 64, 32, 16, 8, 1]
# dropout = 0.0
#
# MODEL_PATH = './model/DNNRegressors/'
# for hl in hidden_layers:
#     MODEL_PATH += '%s_' % hl
# MODEL_PATH += 'D0%s' % (int(dropout * 10))
#
# # Validation and Test Configuration
# validation_metrics = {"MSE": tf.contrib.metrics.streaming_mean_squared_error}
# test_config = skflow.RunConfig(save_checkpoints_steps=100,
#                                save_checkpoints_secs=None)
#
# # Building the Network
# regressor = skflow.DNNRegressor(feature_columns=feature_columns,
#                                 label_dimension=1,
#                                 hidden_units=hidden_layers,
#                                 model_dir=MODEL_PATH,
#                                 dropout=dropout,
#                                 config=test_config)
#
# # Train it
# if TRAINING:
#     print('Train the DNN Regressor...\n')
#     MSEs = []  # for plotting
#     STEPS = []  # for plotting
#
#     for epoch in range(EPOCHS + 1):
#
#         # Fit the DNNRegressor (This is where the magic happens!!!)
#         regressor.fit(input_fn=training_input_fn(batch_size=BATCH_SIZE),
#                       steps=STEPS_PER_EPOCH)
#         # Thats it -----------------------------
#         # Start Tensorboard in Terminal:
#         # 	tensorboard --logdir='./DNNRegressors/'
#         # Now open Browser and visit localhost:6006\
#
#         # This is just for fun and educational purpose:
#         # Evaluate the DNNRegressor every 10th epoch
#         if epoch % 10 == 0:
#             eval_dict = regressor.evaluate(input_fn=test_input_fn(),
#                                            metrics=validation_metrics)
#
#             print('Epoch %i: %.5f MSE' % (epoch + 1, eval_dict['MSE']))
#
#             if WITHPLOT:
#                 # Generate a plot for this epoch to see the Network learning
#                 y_pred = regressor.predict(x={'myInput': total_x}, as_iterable=False)
#
#                 E = (y.reshape((1, -1)) - y_pred)
#                 MSE = np.mean(E ** 2.0)
#                 step = (epoch + 1) * STEPS_PER_EPOCH
#                 title_string = '%s DNNRegressor after %06d steps (MSE=%.5f)' % \
#                                (MODEL_PATH.split('/')[-1], step, MSE)
#
#                 MSEs.append(MSE)
#                 STEPS.append(step)
#
#                 fig = plt.figure(figsize=(9, 4))
#                 ax1 = fig.add_subplot(1, 4, (1, 3))
#                 ax1.plot(total_x, y, label='function to predict')
#                 ax1.plot(total_x, y_pred, label='DNNRegressor prediction')
#                 ax1.legend(loc=2)
#                 ax1.set_title(title_string)
#                 ax1.set_ylim([0, 1])
#
#                 ax2 = fig.add_subplot(1, 4, 4)
#                 ax2.plot(STEPS, MSEs)
#                 ax2.set_xlabel('Step')
#                 ax2.set_xlim([0, EPOCHS * STEPS_PER_EPOCH])
#                 ax2.set_ylabel('Mean Square Error')
#                 ax2.set_ylim([0, 0.01])
#
#                 plt.tight_layout()
#                 plt.savefig(MODEL_PATH + '_%05d.png' % (epoch + 1), dpi=72)
#                 print('Saved %s' % MODEL_PATH + '_%05d.png' % (epoch + 1))
#
#                 plt.close()

# Now it's trained. We can try to predict some values.
# else:
#     print('No training today, just prediction')
#     try:
#         # Prediction
#         X_pred = np.linspace(0, 1, 11)
#         y_pred = regressor.predict(x={'X': X_pred}, as_iterable=False)
#         print(y_pred)
#
#         # Get trained values out of the Network
#         for variable_name in regressor.get_variable_names():
#             if str(variable_name).startswith('dnn/hiddenlayer') and \
#                     (str(variable_name).endswith('weights') or \
#                      str(variable_name).endswith('biases')):
#                 print('\n%s:' % variable_name)
#                 weights = regressor.get_variable_value(variable_name)
#                 print(weights)
#                 print('size: %i' % weights.size)
#
#         # Final Plot
#         if WITHPLOT:
#             plt.plot(X, y, label='function to predict')
#             plt.plot(X, regressor.predict(x={'X': X}, as_iterable=False), \
#                      label='DNNRegressor prediction')
#             plt.legend(loc=2)
#             plt.ylim([0, 1])
#             plt.title('%s DNNRegressor' % MODEL_PATH.split('/')[-1])
#             plt.tight_layout()
#             plt.savefig(MODEL_PATH + '.png', dpi=72)
#             plt.close()
#     except:
#         print('Prediction failed! Maybe first train a model?')
# create model
# model = Sequential()
# model.add(BatchNormalization())
# model.add(Dense(128, input_dim=total_x.shape[1], kernel_initializer='normal'))
# # model.add(BatchNormalization())
# # model.add(Activation("relu"))
# # model.add(Dense(112, kernel_initializer='normal'))
# model.add(BatchNormalization())
# model.add(Activation("relu"))
# model.add(Dense(96, kernel_initializer='normal'))
# # model.add(BatchNormalization())
# # model.add(Activation("relu"))
# # model.add(Dense(80, kernel_initializer='normal'))
# model.add(BatchNormalization())
# model.add(Activation("relu"))
# model.add(Dense(64, kernel_initializer='normal'))
# # model.add(BatchNormalization())
# # model.add(Activation("relu"))
# # model.add(Dense(48, kernel_initializer='normal'))
# model.add(BatchNormalization())
# model.add(Activation("relu"))
# model.add(Dense(32, kernel_initializer='normal'))
# model.add(BatchNormalization())
# model.add(Activation("relu"))
# model.add(Dense(16, kernel_initializer='normal'))
# model.add(BatchNormalization())
# model.add(Activation("relu"))
# model.add(Dense(8, kernel_initializer='normal'))
# model.add(BatchNormalization())
# model.add(Activation("relu"))
# model.add(Dense(1, kernel_initializer='normal'))
# # model.add(Dense(128, input_dim=total_x.shape[1], kernel_initializer='normal', activation="relu"))
# # model.add(Dense(64, kernel_initializer='normal', activation="relu"))
# # model.add(Dense(32, kernel_initializer='normal', activation="relu"))
# # model.add(Dense(16, kernel_initializer='normal', activation="relu"))
# # model.add(Dropout(0.2))
# # model.add(Dense(1, kernel_initializer='normal'))
# # Compile model
# opt = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=10E-8, decay=0.0, amsgrad=False)
# # opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# # opt = RMSprop(lr=0.002, rho=0.9, epsilon=None, decay=0.0)
# model.compile(optimizer=opt, loss='mse', metrics=['mae'])
#
# # 设置模型保存位置
# filepath = "./model/0922/weights-improvement-{epoch:02d}-{loss:.2f}.h5"
# checkpoint = ModelCheckpoint(filepath,
#                              monitor='val_loss',
#                              verbose=1,
#                              save_best_only=True,
#                              save_weights_only=False,
#                              mode='auto')
#
# # 当val_loss不再提升，则停止训练
# earlystop = EarlyStopping(monitor='val_loss',
#                           min_delta=0,
#                           patience=20,
#                           verbose=1,
#                           mode='auto')
#
# tbCallBack = TensorBoard(log_dir='./logs',  # log 目录
#                          histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
#                          write_graph=True,  # 是否存储网络结构图
#                          write_grads=True,  # 是否可视化梯度直方图
#                          write_images=True,  # 是否可视化参数
#                          embeddings_freq=0,
#                          embeddings_layer_names=None,
#                          embeddings_metadata=None)
# callbacks = [checkpoint, earlystop, tbCallBack]
#
# model.fit(total_x, total_y, validation_split=0.15, batch_size=1024, epochs=200, callbacks=callbacks, verbose=1)
#
# model.save("model/my_model.h5")
#
# # 直接测试
# # 原始文件的header顺序
# test_file_path = "E:\\Study\\19华为杯\\赛题\\2019年中国研究生数学建模竞赛A题\\train_set\\train_2915501.csv"
# test_x = []
# test_y = []
# with open(test_file_path, "r") as f:
#     csv_reader = csv.reader(f)
#     next(f)
#     for data_line in csv_reader:
#         one_line_data = np.array(data_line, np.float32)
#         cell_idx = one_line_data[0]
#         cell_x = one_line_data[1]
#         cell_y = one_line_data[2]
#         cell_h = one_line_data[3]
#         azimuth = one_line_data[4]
#         elec_downtilt = one_line_data[5]
#         mechan_downtilt = one_line_data[6]
#         frequency = one_line_data[7]
#         power = one_line_data[8]
#         cell_altitude = one_line_data[9]
#         cell_building_h = one_line_data[10]
#         cell_clutter = int(one_line_data[11])
#         x = one_line_data[12]
#         y = one_line_data[13]
#         altitude = one_line_data[14]
#         building_h = one_line_data[15]
#         clutter = int(one_line_data[16])
#         rsrp = one_line_data[17]
#
#         # 电气特征
#         h_s = cell_h  # 发射机相对地面有效高度
#         horizon_angel = azimuth  # 水平方向角
#         downtilt = elec_downtilt + mechan_downtilt  # 下倾角
#
#         # 几何特征
#         d = math.sqrt((x - cell_x) ** 2 + (y - cell_y) ** 2)  # 水平面上的链路距离
#
#         fire_directrion = np.array(
#             [math.cos(downtilt) * math.sin(horizon_angel), math.cos(downtilt) * math.cos(horizon_angel),
#              -math.sin(downtilt)])
#         bs2receiver_direction = np.array([x - cell_x, y - cell_y, (cell_h + cell_altitude) - altitude])
#         # 信号线与实际信号之间的夹角的cos值，考虑分母为零的情况报错
#         beta_cos = fire_directrion.dot(bs2receiver_direction) / (
#                 np.sqrt(fire_directrion.dot(fire_directrion)) * np.sqrt(
#             bs2receiver_direction.dot(bs2receiver_direction)))
#
#         # 接收点到信号线的垂直距离
#         h_v = math.sqrt(
#             (x - cell_x) ** 2 + (y - cell_y) ** 2 + (cell_h + cell_altitude - altitude) ** 2) * math.sqrt(
#             1 - beta_cos ** 2)
#
#         # 环境特征
#         # 分七个类
#         classes = {
#             "10": 0, "11": 0, "12": 0, "20": 0, "13": 1, "14": 1, "16": 1, "15": 2, "18": 2,
#             "17": 3, "19": 3, "1": 4, "2": 4, "3": 4, "4": 5, "5": 5, "6": 5, "7": 6, "8": 6, "9": 6
#         }
#         cell_clutter_vec = [0] * 7  # 发射机所在地物特征
#         clutter_vec = [0] * 7  # 接收机所在地物特征
#
#         cell_clutter_vec[classes[str(cell_clutter)]] = 1
#         clutter_vec[classes[str(clutter)]] = 1
#
#         h_build_s = cell_building_h
#         h_build_r = building_h
#         alti_diff = cell_altitude - altitude
#
#         # 不可训练特征
#         p_t = power
#         f = frequency
#
#         # 构造feature list
#         feature_list = \
#             [
#                 # 电器特征
#                 power,  # 发射功率
#                 math.log(abs(h_s + 1)),
#                 horizon_angel,
#                 downtilt,
#                 math.log(f),  # 先放进去试试
#                 # 几何特征
#                 math.log(d / 1000 + 1),
#                 beta_cos,
#                 math.log(h_v / 1000 + 1),
#                 math.log(d / 1000 + 1) * math.log(abs(h_s + 1)),
#                 # 环境特征
#                 h_build_s,
#                 h_build_r,
#                 alti_diff,
#                 round(cell_idx / 100)  # 小区去掉最后两位
#             ]
#         feature_list.extend(cell_clutter_vec)
#         feature_list.extend(clutter_vec)
#
#         features = np.asarray(feature_list, dtype=np.float32)
#         test_x.append(features)
#         test_y.append(rsrp)
#
# test_x = np.array(test_x)
# test_y = np.array(test_y)
# # 测h5模型对不对
#
# pred_y = model.predict(test_x)
# print(pred_y)
