import csv
import os
import pickle

import math

dir_path = "E:\Study\\19华为杯\赛题\\2019年中国研究生数学建模竞赛A题\\train_set"
file_list = os.listdir(dir_path)

print("file number: " + str(len(file_list)))

# with open('./data/whole_data.pkl', 'rb') as f:
#     whole_data = pickle.load(f)


for i in range(len(file_list)):
    file = file_list[i]
    print("handling No.%04d file: %s" % (i + 1, file))

    file_path = os.path.join(dir_path, file)

    file_data = []

    with open(file_path, "r") as f:
        file_reader = csv.DictReader(f)
        idx = 2
        for data_line in file_reader:
            # 取出各项数据，转换成合适的格式
            cell_index = int(data_line["Cell Index"])
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

            # 去除异常数据
            d = math.sqrt(((x - cell_x) * 5) ** 2 + ((y - cell_y) * 5) ** 2) / 1000
            if d > 15 and rsrp > -95:
                print("异常数据：" + str(idx))
                print("d: %f || rsrp: %f || cell height: %f || height: %f || cell clutter: %d || clutter %d" %
                      (d, rsrp, cell_h + cell_building_h + cell_altitude, building_h + altitude, cell_clutter, clutter))
            # 类型转换好的数据
            # data_dict = {}
            # data_dict["cell_index"] = cell_index
            # data_dict["cell_x"] = cell_x
            # data_dict["cell_y"] = cell_y
            # data_dict["cell_h"] = cell_h
            # data_dict["cell_building_h"] = cell_building_h
            # data_dict["cell_altitude"] = cell_altitude
            # data_dict["cell_clutter"] = cell_clutter
            # data_dict["azimuth"] = azimuth
            # data_dict["elec_downtilt"] = elec_downtilt
            # data_dict["mechan_downtilt"] = mechan_downtilt
            # data_dict["frequency"] = frequency
            # data_dict["power"] = power
            # data_dict["x"] = x
            # data_dict["y"] = y
            # data_dict["building_h"] = building_h
            # data_dict["altitude"] = altitude
            # data_dict["clutter"] = clutter
            # data_dict["rsrp"] = rsrp
            #
            # file_data.append(data_dict)
            idx += 1
# with open('./data/whole_data.pkl', 'wb') as f:
#     pickle.dump(whole_data, f)
