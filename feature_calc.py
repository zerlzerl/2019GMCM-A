import csv
import os
import numpy as np
import math


def data_preprocess(data_line):
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
    beta_cos = fire_directrion.dot(bs2receiver_direction) / (np.sqrt(fire_directrion.dot(fire_directrion)) * np.sqrt(
        bs2receiver_direction.dot(bs2receiver_direction)))

    # 接收点到信号线的垂直距离
    h_v = math.sqrt((x - cell_x) ** 2 + (y - cell_y) ** 2 + (cell_h + cell_altitude - altitude) ** 2) * math.sqrt(
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

    # 标签
    rsrp = rsrp

    # # 构造feature list
    # feature_list = \
    #     [
    #         # 电器特征
    #         h_s,
    #         horizon_angel,
    #         downtilt,
    #         # 几何特征
    #         d,
    #         beta_cos,
    #         h_v,
    #         # 环境特征
    #         cell_clutter,  # 还是先存类别信息，占用空间少
    #         clutter,
    #         h_build_s,
    #         h_build_r,
    #         alti_diff,
    #         # 不可训练
    #         p_t,
    #         f,
    #         # 标签
    #         rsrp
    #     ]
    # return feature_list

    # 构造feature字典
    feature_dict = \
        {
            # 电器特征
            "cell_h": h_s,
            "azimuth": horizon_angel,
            "downtilt": downtilt,
            # 几何特征
            "d": d,
            "beta_cos": beta_cos,
            "h_v": h_v,
            # 环境特征
            "cell_clutter": cell_clutter,  # 还是先存类别信息，占用空间少
            "clutter": clutter,
            "h_build_s": h_build_s,
            "h_build_r": h_build_r,
            "alti_diff": alti_diff,
            # 不可训练
            "power": p_t,
            "frequency": f,
            # 标签
            "rsrp": rsrp
        }
    return feature_dict

feature_header = \
    [
        # 电器特征
        "cell_h",
        "azimuth",
        "downtilt",
        # 几何特征
        "d",
        "beta_cos",
        "h_v",
        # 环境特征
        "cell_clutter",  # 还是先存类别信息，占用空间少
        "clutter",
        "h_build_s",
        "h_build_r",
        "alti_diff",
        # 不可训练
        "power",
        "frequency",
        # 标签
        "rsrp"
    ]

cleaned_data_dir = "E:\\Study\\19华为杯\\赛题\\2019年中国研究生数学建模竞赛A题\\cleaned_train_set\\"
featured_data_dir = "E:\\Study\\19华为杯\\赛题\\2019年中国研究生数学建模竞赛A题\\featured_train_set\\"
file_list = os.listdir(cleaned_data_dir)

for file in file_list:
    file_path = os.path.join(cleaned_data_dir, file)
    featured_file_path = os.path.join(featured_data_dir, file)
    with open(file_path, "r") as f:
        with open(featured_file_path, "w", newline='') as wf:
            csv_reader = csv.DictReader(f)
            csv_writer = csv.DictWriter(wf, fieldnames=feature_header)
            csv_writer.writeheader()
            for row in csv_reader:
                try:
                    features = data_preprocess(row)
                    csv_writer.writerow(features)
                except:
                    print("error occur.")
