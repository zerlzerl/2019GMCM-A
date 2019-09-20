import csv

import matplotlib.pyplot as plt
x = list(range(0,100))

test_y = []
pred_y = []
with open("./data/result/result.csv") as f:
    csv_reader = csv.DictReader(f)
    for row in csv_reader:
        test_y.append(row["test_rsrp"])
        pred_y.append(row["pred_rsrp"])


plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')
plt.show()