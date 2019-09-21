import csv
import numpy as np

import matplotlib.pyplot as plt
x = np.arange(0, 100)

test_y = []
pred_y = []
with open("./data/result/result.csv") as f:
    csv_reader = csv.DictReader(f)
    for row in csv_reader:
        test_y.append(float(row["test_rsrp"]))
        pred_y.append(float(row["pred_rsrp"]))

tp = 0
fp = 0
fn = 0
tn = 0
for i in range(len(test_y)):
    if test_y[i] < -103 and pred_y[i] < -103:
        tp += 1
    if test_y[i] < -103 and pred_y[i] > -103:
        fn += 1
    if test_y[i] > -103 and pred_y[i] < -103:
        fp += 1
    if test_y[i] > -103 and pred_y[i] > -103:
        tn += 1

precision = tp / (tp + fp)
print("precision:" + str(precision))
recall = tp / (tp + fn)
print("recall:" + str(recall))
pcrr = 2 * precision * recall / (precision + recall)
print("PCRR:" + str(pcrr))

msre = np.sqrt(((np.array(test_y) - np.array(pred_y)) ** 2).mean())
print("MSRE:" + str(msre))

plt.plot(x,test_y[:100], 'r-', label='test_y')
plt.plot(x,pred_y[:100], 'g-', label='predict_y')
plt.xlabel('index')
plt.ylabel('rsrp')
plt.legend()
plt.show()