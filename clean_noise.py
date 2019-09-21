import csv
import os
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth, AgglomerativeClustering, DBSCAN
import numpy as np

dir_path = "E:\Study\\19华为杯\赛题\\2019年中国研究生数学建模竞赛A题\\train_set"
file_list = os.listdir(dir_path)

print("file number: " + str(len(file_list)))
for i in range(len(file_list)):
    file = file_list[i]
    print("handling No.%04d file: %s" % (i + 1, file))
    file_path = os.path.join(dir_path, file)

    receiver_position = []
    cell_position = []
    data_dict = {}
    with open(file_path, "r") as f:
        csv_reader = csv.DictReader(f)
        idx = 0
        for data_line in csv_reader:
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

            if len(cell_position) == 0:
                cell_position = [cell_x, cell_y]

            receiver_position.append([x, y])

            data_dict[str(idx)] = data_line
            idx += 1

    # 聚类
    bandwidth = estimate_bandwidth(receiver_position, quantile=0.2)
    # clustering = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    clustering = AgglomerativeClustering(linkage="single", n_clusters=5)
    # clustering = DBSCAN(eps=10, min_samples=10)
    clustering.fit(receiver_position)

    labels = clustering.labels_
    # cluster_centers = clustering.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    pos_statistic = {}
    for i in range(len(labels)):
        label = labels[i]
        if str(label) in pos_statistic:
            group = pos_statistic[str(label)]
            group.append(receiver_position[i])
            pos_statistic[str(label)] = group
        else:
            group = [receiver_position[i]]
            pos_statistic[str(label)] = group

    colors = ['b', 'g', 'c', 'm', 'y', 'k',
              'b', 'g', 'c', 'm', 'y', 'k',
              'b', 'g', 'c', 'm', 'y', 'k']

    sender_position = cell_position
    cleaned_data = []
    noise_data = []
    threshold = 3000
    for key in pos_statistic:
        group = np.array(pos_statistic[key])
        center_position = np.mean(group, axis=0)
        center_distance = np.linalg.norm(center_position - sender_position)
        if center_distance < threshold and len(group) > 10:
            cleaned_data.append(group)
            plt.scatter(group[:, 0], group[:, 1], color=colors[int(key)], label=key)
        else:
            noise_data.append(group)

    plt.scatter(cell_position[0], cell_position[1], marker='v', color="r", alpha=0.7)

    plt.show()

    # 去除噪声数据
    num_noise = len(noise_data)
    num_clean = len(cleaned_data)
    total = len(data_dict)
    print("noise data: %d, %.2f || clean data: %d, %.2f" % (num_noise, num_noise/total, num_clean, num_clean/total))
    # min_dist_key = min(cluster_center_distance, key=cluster_center_distance.get)
    # min_dist_group_member = pos_statistic[min_dist_key]
    # print("member number: %d, rate: %.2f" % (len(min_dist_group_member), len(min_dist_group_member) / len(data_dict)))
