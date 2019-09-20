# 从csv读取原始数据
import os
import csv


def read_dir(dir_path):
    files = os.listdir(dir_path)
    data_list = []
    for file in files:
        if not os.path.isdir(file):
            file_path = os.path.join(dir_path, file)
            print(file_path)
            with open(file_path, "r") as f:
                file_reader = csv.DictReader(f)
                for row in file_reader:
                    data_list.append(row)
    return data_list


def read_training_data():
    training_dir = "./data/sample/train/"
    return read_dir(training_dir)


def read_test_data():
    test_dir = "./data/sample/test/"
    return read_dir(test_dir)


if __name__ == '__main__':
    read_training_data()
