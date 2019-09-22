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

                    cell_clutter_vec = [0] * 20
                    clutter_vec = [0] * 20
                    cell_clutter_vec[cell_clutter - 1] = 1
                    clutter_vec[clutter - 1] = 1

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
                        ]
                    feature_list.extend(cell_clutter_vec)
                    feature_list.extend(clutter_vec)

                    features = np.asarray(feature_list, dtype=np.float32)
                    featured_data.append(features)

                print(file_name, len(featured_data))
                filesDatas.append(np.array(featured_data))

        filesDatas = np.array(filesDatas, dtype=np.float32).reshape(-1, 52)
        preprocessed_data['myInput'] = filesDatas
        print("preprocessed_data[\'myInput\'].shape = ", preprocessed_data['myInput'].shape)

        return preprocessed_data

    def _postprocess(self, data):
        infer_output = {"RSRP": []}
        for output_name, results in data.items():
            print(output_name, np.array(results).shape)
            infer_output["RSRP"] = results

        return infer_output