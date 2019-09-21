import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
from data_reader import *
from data_preprocess import *

# train = pd.read_csv('./train_108401.csv')
preprocessed_training_data = []
label = []
training_data = read_training_data()
for data_line in training_data:
    x, y = data_preprocess(data_line)
    x = np.append(x, float(y))
    preprocessed_training_data.append(x)
    # label.append(y)

corrmat = np.corrcoef(preprocessed_training_data, rowvar=0)
# corrmat.to_csv('corrmat.csv')
plt.subplots()
sns.heatmap(data = corrmat,
            # cmap="RdPu",
            vmin=-1,
            vmax=1,
            fmt='d',
            linewidths=.5,
            square = True)

plt.show()