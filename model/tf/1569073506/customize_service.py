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
                    cell_index = one_line_data[0]
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
                    # rsrp = one_line_data[17]

                    rel_x = x - cell_x
                    rel_y = y - cell_y
                    rel_h = (cell_h + cell_building_h + cell_altitude) - (building_h + altitude)

                    d = math.sqrt(
                        math.pow(rel_x * 5, 2) + math.pow(rel_y * 5, 2) + math.pow(rel_h, 2))

                    downtilt = elec_downtilt + mechan_downtilt
                    fire_directrion = np.array(
                        [math.cos(downtilt) * math.sin(azimuth), math.cos(downtilt) * math.cos(azimuth),
                         -math.sin(downtilt)])
                    bs2receiver_direction = np.array([rel_x, rel_y, -rel_h])
                    angle_cos = fire_directrion.dot(bs2receiver_direction) / (
                                np.sqrt(fire_directrion.dot(fire_directrion)) * np.sqrt(
                            bs2receiver_direction.dot(bs2receiver_direction)))

                    cell_clutter_vec = [0] * 20
                    clutter_vec = [0] * 20

                    cell_clutter_vec[cell_clutter - 1] = 1
                    clutter_vec[clutter - 1] = 1

                    train_vec = np.array([d, rel_h, angle_cos, power], dtype=np.float)
                    train_vec = np.append(train_vec, cell_clutter_vec)
                    train_vec = np.append(train_vec, clutter_vec)

                    featured_data.append(train_vec)

                print(file_name, len(featured_data))
                filesDatas.append(np.array(featured_data))

        filesDatas = np.array(filesDatas, dtype=np.float32).reshape(-1, 17)
        preprocessed_data['myInput'] = filesDatas
        print("preprocessed_data[\'myInput\'].shape = ", preprocessed_data['myInput'].shape)

        return preprocessed_data


    def _postprocess(self, data):
        infer_output = {"RSRP": []}
        for output_name, results in data.items():
            print(output_name, np.array(results).shape)
            infer_output["RSRP"] = results
            # output_file_name = output_name + "_result.txt"
            # with open(output_file_name, "w") as f:
            #     f.write(str(infer_output))

        return infer_output