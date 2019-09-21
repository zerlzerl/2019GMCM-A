import os
import csv

dir_path = "../MathModel/train_set/"


# dir_path = "./data/sample/train/"

def read_file(path):
    res = {}
    with open(path, "r", encoding='UTF-8') as f:
        file_reader = csv.DictReader(f)
        for row in file_reader:
            pos = (int(row["X"]), int(row["Y"]))
            type = int(row["Clutter Index"])
            if res.get(pos) == None:
                res[pos] = type
            elif res.get(pos) != type:
                print(str(path) + ': ' + str(pos) + ' ' + str(type) + ' ' + str(res.get(pos)))
    # 返回字典 (x, y)->type
    return res


files = os.listdir(dir_path)
num = len(files)
# 分别统计交集数量 和 交集中相同地物类型数量
# res1, res2 = [], []
res = []
f = open('matches.txt', 'w')  # 清空文件内容再写

for i in range(num):
    dict1 = read_file(os.path.join(dir_path, files[i]))
    # cur1, cur2 = [], []
    for j in range(i + 1, num):
        tmp1, tmp2 = 0, 0
        dict2 = read_file(os.path.join(dir_path, files[j]))

        for x in dict1:
            if x in dict2:
                tmp1 = tmp1 + 1
                if dict1[x] == dict2[x]:
                    tmp2 = tmp2 + 1
        if tmp1 > 0:
            print(str(files[i]) + "和" + str(files[j]) + "交集有" + str(tmp1) + ", 相同有" + str(tmp2))
            print("匹配率为" + str(float(tmp2) / float(tmp1)))
            f.write(str(files[i]) + '\t' + str(files[j]) + '\t' + str(tmp2 // tmp1) + '\n')
        # cur1.append(tmp1)
        # cur2.append(tmp2)
    # res1.append(cur1)
    # res2.append(cur2)
f.close()