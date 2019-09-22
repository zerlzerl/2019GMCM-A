import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
# import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
from data_reader import *
from data_preprocess import *

# train = pd.read_csv('./train_108401.csv')
# preprocessed_training_data = []
# label = []
# training_data = read_training_data()
# for data_line in training_data:
# x, y = data_preprocess(data_line)
# x = np.append(x, float(y))
# preprocessed_training_data.append(x)
# label.append(y)

# corrmat = np.corrcoef(preprocessed_training_data, rowvar=0)
# corrmat.to_csv('corrmat.csv')
# plt.subplots()
# sns.heatmap(data = corrmat,
#             # cmap="RdPu",
#             vmin=-1,
#             vmax=1,
#             fmt='d',
#             linewidths=.5,
#             square = True)
#
# plt.show()

# dir_path = './data/sample/train/'
dir_path = './data/featured_train_set/'


def read_from_origin():
    train = pd.DataFrame()

    n = 0
    files = os.listdir(dir_path)
    for file in files:
        n = n + 1
        if n % 100 == 0:
            print(n)
        # if n == 2:
        #     break
        if not os.path.isdir(file):
            file_path = os.path.join(dir_path, file)
            # print(file_path)
            tmp = pd.read_csv(file_path).sample(frac=0.05)
            train = train.append(tmp, ignore_index=True)

    # 丢弃Nan项
    train.dropna()

    # 对数变换
    # for key in train.keys():
    #     if key not in ["rsrp", "cell_clutter", "clutter", "power","RSRP", "Cell Clutter Index", "Clutter Index"]:
    #         train[key] = np.log10(np.abs(train[key]))

    for key in train.keys():
        if key in ["frequency", "cell_h", "h_v", "d", "s"]:
            train[key] = np.log10(np.abs(train[key] + 0.000001))
    # train["rsrp"] = train["rsrp"] - train["power"]

    # 添加列 logd * logh
    train.insert(13, "logh*logd", train["h_v"] * train["d"])
    train.to_csv('test.csv', index=0)
    return train


if __name__ == "__main__":
    if not os.path.exists('test.csv'):
        train = read_from_origin()
    else:
        train = pd.read_csv('test.csv')
    corrmat = train.corr('spearman')
    corrmat.to_csv('corrmat_featured_spearman.csv')
    # plt.subplots()
    # plt.figure(figsize=(20, 20))
    sns.heatmap(corrmat,
                vmax=.5,
                vmin=-.5,
                cmap='rocket_r',
                # cmap='YlGnBu',
                center=0,
                square=True,
                # robust=True,
                # square=True,
                # annot=True
                )
    # plt.show()
