import csv
import os
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import numpy as np

dir_path = "E:\Study\\19华为杯\赛题\\2019年中国研究生数学建模竞赛A题\\train_set"
file_list = os.listdir(dir_path)

print("file number: " + str(len(file_list)))
for i in range(len(file_list)):
    file = file_list[i]
    # file = "train_1280001.csv"
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

            data_dict[",".join([str(x), str(y)])] = data_line
            idx += 1

    # 聚类
    # bandwidth = estimate_bandwidth(receiver_position, quantile=0.2)
    # clustering = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    clustering = AgglomerativeClustering(linkage="single", n_clusters=25)
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
    distance_set = []
    threshold = 3000
    for key in pos_statistic:
        group = np.array(pos_statistic[key])
        center_position = np.mean(group, axis=0)
        center_distance = np.linalg.norm(center_position - sender_position)
        distance_set.append(center_distance)

    # 计算distance的均值和方差
    dist_mean = np.array(distance_set).mean()
    dist_std = np.array(distance_set).std()
    threshold = dist_mean + 3 * dist_std

    for key in pos_statistic:
        group = np.array(pos_statistic[key])
        center_position = np.mean(group, axis=0)
        center_distance = np.linalg.norm(center_position - sender_position)
        if center_distance < threshold and len(group) > 10:
            cleaned_data.extend(group.tolist())
            # plt.scatter(group[:, 0], group[:, 1], color=colors[int(key)], label=key)
        else:
            noise_data.extend(group.tolist())

    # plt.scatter(cell_position[0], cell_position[1], marker='v', color="r", alpha=0.7)
    #
    # plt.show()

    # 去除噪声数据
    num_noise = len(noise_data)
    num_clean = len(cleaned_data)
    total = len(receiver_position)
    print("noise data: %d, %.2f || clean data: %d, %.2f" % (num_noise, num_noise/total, num_clean, num_clean/total))

    cleaned_data_list = []
    for data_pos in cleaned_data:
        key = ",".join([str(data_pos[0]), str(data_pos[1])])
        if key in data_dict:
            cleaned_data_list.append(data_dict[key])

    # 存入到csv
    with open("E:\Study\\19华为杯\\赛题\\2019年中国研究生数学建模竞赛A题\\cleaned_train_set\\"+ file.split('.')[0] + "_cleaned.csv", "w", newline="") as f:
        csv_writer = csv.DictWriter(f, cleaned_data_list[0].keys())
        csv_writer.writeheader()
        for line_dict in cleaned_data_list:
            csv_writer.writerow(line_dict)
    # 可视化
    fig = plt.figure(figsize=(20, 8))
    # plt.title("Abnormal Data Processing")

    # 清理前
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("Original")
    ax1.scatter(np.array(receiver_position)[:, 0], np.array(receiver_position)[:, 1], c='b', marker='^', s=1, alpha=.6)
    ax1.scatter(np.array(noise_data)[:, 0], np.array(noise_data)[:, 1], c='r', marker='x', s=2, alpha=.6)
    ax1.scatter(cell_position[0], cell_position[1], c='r', s=100, marker='o')

    # 清理后
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("Cleaned")
    ax2.scatter(np.array(cleaned_data)[:, 0], np.array(cleaned_data)[:, 1], c='b', marker='^', s=1, alpha=.6)
    ax2.scatter(cell_position[0], cell_position[1], c='r', s=100, marker='o')

    # plt.legend()
    plt.savefig("E:\Study\\19华为杯\赛题\\2019年中国研究生数学建模竞赛A题\\25_cluster_adjust_10\\" + file.split('.')[0] + '.png', dpi=400, bbox_inches='tight')
    plt.close()
    # min_dist_key = min(cluster_center_distance, key=cluster_center_distance.get)
    # min_dist_group_member = pos_statistic[min_dist_key]
    # print("member number: %d, rate: %.2f" % (len(min_dist_group_member), len(min_dist_group_member) / len(data_dict)))

