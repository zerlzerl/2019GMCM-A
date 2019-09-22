import csv
import os

dir_path = "E:\Study\\19华为杯\赛题\\2019年中国研究生数学建模竞赛A题\\cleaned_train_set"
file_list = os.listdir(dir_path)

print("file number: " + str(len(file_list)))
weak_total = 0
total = 0
for i in range(len(file_list)):
    file = file_list[i]
    # file = "train_1280001.csv"
    file_path = os.path.join(dir_path, file)

    weak = 0
    file_total = 0
    with open(file_path, "r") as f:
        csv_reader = csv.DictReader(f)
        idx = 0
        for data_line in csv_reader:
            file_total += 1
            total += 1
            # 取出各项数据，转换成合适的格式
            rsrp = float(data_line["RSRP"])
            if rsrp < -103:
                weak += 1
                weak_total += 1

    print("handling No.%04d file: %s || weak: %d , %.3f" % (i + 1, file, weak, weak / file_total))

print("total PCRR: %d, %.4f" % (weak_total, weak_total / total))
