import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
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
dir_path = './data/cleaned_train_set/'

train = pd.DataFrame()

n = 0
files = os.listdir(dir_path)
for file in files:
    n = n + 1
    if n % 100 == 0:
        print(n)
    # if n == 1000:
    #     break
    if not os.path.isdir(file):
        file_path = os.path.join(dir_path, file)
        # print(file_path)
        tmp = pd.read_csv(file_path).sample(frac=0.01)
        train = train.append(tmp, ignore_index=True)

corrmat = train.corr()
corrmat.to_csv('corrmat.csv')
plt.subplots()
sns.heatmap(corrmat,
            # vmax=1,
            # vmin=0,
            cmap='rocket_r',
            # robust=True,
            square=True)
plt.show()
