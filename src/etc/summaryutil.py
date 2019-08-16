import os
import csv
import collections
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-colorblind')
p2d = '../../results/06_07_2017/VET/2003'

ddata = collections.defaultdict(list)

total = collections.defaultdict(int)
invalid_precision = collections.defaultdict(int)
invalid_recall = collections.defaultdict(int)

for subdir in os.listdir(p2d):
    subdir = os.path.join(p2d, subdir)
    if os.path.isdir(subdir):
        n_ls = int(os.path.basename(subdir).split('_')[-2])
        with open(os.path.join(subdir, 'summary_report.csv')) as f:
            reader = csv.reader(f)
            reader.__next__()
            d = []
            for row in reader:
                d.append(list(map(float, row)))
            d = np.array(d)

        d = d[:, [0, 3, 4]]

        total[n_ls] += (d.shape[0] - 1)
        invalid_precision[n_ls] += np.where(d[:, 1] != d[:, 1])[0].size
        invalid_recall[n_ls] += np.where(d[:, 2] != d[:, 2])[0].size

        s = d[-1]
        s[0] -= 1
        s[0] /= 200
        ddata[n_ls].append(s)

data = []
sum_csv = []

for n_ls, d in sorted(ddata.items(), key=lambda tup: tup[0]):
    d = np.array(d)
    data.append(d)
    min_ = d[1:].min(axis=0)
    mean_ = d[1:].mean(axis=0)
    max_ = d[1:].max(axis=0)
    d = np.c_[min_, mean_, max_].ravel()
    sum_csv.append(d)

sum_csv = np.c_[np.arange(5, 25, 5), np.round(sum_csv, 2)].astype(str)
# header = ['Listeners', 'Discovery Rate', 'Precision', 'Recall']

with open('../../results/summary.csv', 'w') as f:
    writer = csv.writer(f)
    # writer.writerow(header)
    writer.writerows(sum_csv)

data = np.array(data)
dr = data[:, :, 0]
p = data[:, :, 1]
r = data[:, :, 2]

x = np.arange(5, 25, 5)
y = np.array([dr.mean(axis=1), p.mean(axis=1), r.mean(axis=1)])
yerr = np.array([np.std(dr, axis=1), np.std(p, axis=1), np.std(r, axis=1)])

fig, ax = plt.subplots()

for y_, yerr_, lbl in zip(y, yerr, ['discovery rate', 'precision', 'recall']):
    ax.errorbar(x, y_, yerr=yerr_, capsize=2, label=lbl)

box = ax.get_position()
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=3)
ax.set_xlabel('Number of Listeners')

ax.set_xticks(x)
plt.xlim(4, 21)
plt.ylim(0, 1.1)
plt.savefig('../../results/summary', dpi=500)


data = []

for n_ls in [5, 10, 15, 20]:
    t = total[n_ls] / 100
    ip = invalid_precision[n_ls] / 100
    ir = invalid_recall[n_ls] / 100
    ipr = ip / t
    irr = ir / t

    data.append([n_ls, t, ip, ipr, ir, irr])

data = np.round(data, 2).astype(str)

with open('../../results/summary_nan.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(data)