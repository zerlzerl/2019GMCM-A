import math
import numpy as np
from model_service.tfserving_model_service import TfServingBaseService
import pandas as pd


class mnist_service(TfServingBaseService):
    def _preprocess(self, data):
        preprocessed_data = {}
        filesDatas = []
        for k, v in data.items():
            for file_name, file_content in v.items():  # csv file name, csv content
                pb_data = pd.read_csv(file_content)
                input_data = np.array(pb_data.get_values()[:, 0:17], dtype=np.float32)  # 0-17
                featured_data = []
                for i in range(input_data.shape[0]):
                    one_line_data = input_data[i, :]

                    cell_idx = one_line_data[0]
                    if np.isnan(cell_idx):
                        if i == 0:
                            cell_idx = 1668101.0
                        else:
                            cell_idx = input_data[i-1, 0]

                    cell_x = one_line_data[1]
                    if np.isnan(cell_x):
                        if i == 0:
                            cell_x = 407180.0
                        else:
                            cell_x = input_data[i - 1, 1]

                    cell_y = one_line_data[2]
                    if np.isnan(cell_y):
                        if i == 0:
                            cell_y = 3396265.0
                        else:
                            cell_y = input_data[i - 1, 2]

                    cell_h = one_line_data[3]
                    if np.isnan(cell_h):
                        if i == 0:
                            cell_h = 0.0
                        else:
                            cell_h = input_data[i - 1, 3]

                    azimuth = one_line_data[4]
                    if np.isnan(azimuth):
                        if i == 0:
                            azimuth = 0.0
                        else:
                            azimuth = input_data[i - 1, 4]

                    elec_downtilt = one_line_data[5]
                    if np.isnan(elec_downtilt):
                        if i == 0:
                            elec_downtilt = 0.0
                        else:
                            elec_downtilt = input_data[i - 1, 5]

                    mechan_downtilt = one_line_data[6]
                    if np.isnan(mechan_downtilt):
                        if i == 0:
                            mechan_downtilt = 0.0
                        else:
                            mechan_downtilt = input_data[i - 1, 6]

                    frequency = one_line_data[7]
                    if np.isnan(frequency):
                        if i == 0:
                            frequency = 2585.0
                        else:
                            frequency = input_data[i - 1, 7]

                    power = one_line_data[8]
                    if np.isnan(power):
                        if i == 0:
                            power = 10.0
                        else:
                            power = input_data[i - 1, 8]

                    cell_altitude = one_line_data[9]
                    if np.isnan(cell_altitude):
                        if i == 0:
                            cell_altitude = 500
                        else:
                            cell_altitude = input_data[i - 1, 9]

                    cell_building_h = one_line_data[10]
                    if np.isnan(cell_building_h):
                        if i == 0:
                            cell_building_h = 0.0
                        else:
                            cell_building_h = input_data[i - 1, 10]

                    cell_clutter = one_line_data[11]
                    if np.isnan(cell_clutter):
                        if i == 0:
                            cell_clutter = 5
                        else:
                            cell_clutter = input_data[i - 1, 11]
                    else:
                        cell_clutter = int(one_line_data[11])

                    x = one_line_data[12]
                    if np.isnan(x):
                        if i == 0:
                            x = cell_x + 500
                        else:
                            x = input_data[i - 1, 12]

                    y = one_line_data[13]
                    if np.isnan(y):
                        if i == 0:
                            y = cell_y + 500
                        else:
                            y = input_data[i - 1, 13]

                    altitude = one_line_data[14]
                    if np.isnan(altitude):
                        if i == 0:
                            altitude = cell_altitude
                        else:
                            altitude = input_data[i - 1, 14]

                    building_h = one_line_data[15]
                    if np.isnan(building_h):
                        if i == 0:
                            building_h = 0.0
                        else:
                            building_h = input_data[i - 1, 15]

                    clutter = one_line_data[16]
                    if np.isnan(clutter):
                        if i == 0:
                            clutter = 6
                        else:
                            clutter = input_data[i - 1, 16]
                    else:
                        clutter = int(one_line_data[16])
                    h_s = cell_h
                    horizon_angel = azimuth
                    downtilt = elec_downtilt + mechan_downtilt


                    d = math.sqrt((x - cell_x) ** 2 + (y - cell_y) ** 2)

                    fire_directrion = np.array(
                        [math.cos(downtilt) * math.sin(horizon_angel), math.cos(downtilt) * math.cos(horizon_angel),
                         -math.sin(downtilt)])
                    bs2receiver_direction = np.array([x - cell_x, y - cell_y, (cell_h + cell_altitude) - altitude])
                    beta_cos = fire_directrion.dot(bs2receiver_direction) / (
                            np.sqrt(fire_directrion.dot(fire_directrion)) * np.sqrt(
                        bs2receiver_direction.dot(bs2receiver_direction)))

                    h_v = math.sqrt(
                        (x - cell_x) ** 2 + (y - cell_y) ** 2 + (cell_h + cell_altitude - altitude) ** 2) * math.sqrt(
                        1 - beta_cos ** 2)

                    # 分七个类
                    classes = {
                        "10": 0, "11": 0, "12": 0, "20": 0, "13": 1, "14": 1, "16": 1, "15": 2, "18": 2,
                        "17": 3, "19": 3, "1": 4, "2": 4, "3": 4, "4": 5, "5": 5, "6": 5, "7": 6, "8": 6, "9": 6
                    }
                    cell_clutter_vec = [0] * 7  # 发射机所在地物特征
                    clutter_vec = [0] * 7  # 接收机所在地物特征

                    cell_clutter_vec[classes[str(int(cell_clutter))]] = 1
                    clutter_vec[classes[str(int(clutter))]] = 1

                    h_build_s = cell_building_h
                    h_build_r = building_h
                    alti_diff = cell_altitude - altitude

                    f = frequency

                    feature_list = \
                        [
                            power,
                            math.log(abs(h_s + 1)),
                            horizon_angel,
                            downtilt,
                            math.log(f),
                            math.log(d / 1000 + 1),
                            beta_cos,
                            math.log(h_v / 1000 + 1),
                            math.log(d / 1000 + 1) * math.log(abs(h_s + 1)),
                            h_build_s,
                            h_build_r,
                            alti_diff,
                            round(cell_idx / 100)  # 小区去掉最后两位
                        ]
                    feature_list.extend(cell_clutter_vec)
                    feature_list.extend(clutter_vec)

                    features = np.asarray(feature_list, dtype=np.float32)
                    featured_data.append(features)

                print(file_name, len(featured_data))
                filesDatas.append(np.array(featured_data))

        filesDatas = np.array(filesDatas, dtype=np.float32).reshape(-1, 27)
        preprocessed_data['myInput'] = filesDatas
        print("preprocessed_data[\'myInput\'].shape = ", preprocessed_data['myInput'].shape)

        return preprocessed_data

    def _postprocess(self, data):
        infer_output = {"RSRP": []}
        for output_name, results in data.items():
            print(output_name, np.array(results).shape)
            infer_output["RSRP"] = results

        return infer_output