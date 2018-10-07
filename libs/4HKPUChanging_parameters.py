#All training with/with out considering class=150 in line=67
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def get_feature(vector, features):
    return [
        f for f, v in zip(features, vector) if v == 1
    ]


def get_class(y):
    Chot = np.array([0.0] * 193)
    Chot[int(y)] = 1.0
    return Chot


def all_zero(col):
    for elem in col:
        if elem != 0:
            return False
    return True


def get_file(n, j):
    train = []
    test = []
    for i in range(193):
        ds = data[data['label'] == i]
        dtr = ds.iloc[:n]
        dte = ds.iloc[n:n + j]
        for value in dtr.values:
            train.append(value)
        for value in dte.values:
            test.append(value)

    train = pd.DataFrame(data=train, columns=[str(i) for i in range(4096)] + ['label'])
    test = pd.DataFrame(data=test, columns=[str(i) for i in range(4096)] + ['label'])
    return train, test


data = pd.read_csv('final_features.csv')
con = {1: 4, 2: 4, 3:4, 4: 4, 5: 4, 6: 4, 7: 4, 8:4,9:4,10:4,11:4,12:4,13:4,14:4,15:4}
sc = []
sc1 = []
sc2 = []
fig = plt.figure()
#LinearSVC(C=1,)
#kernel='poly',max_iter=2000
#SVC(C=1.0,)
models = [LinearSVC(C=1,), RandomForestClassifier(n_estimators=200), KNeighborsClassifier(n_neighbors=3)]
for n in range(1, 16):
    for j in range(1,2):
        print 'train:', n, 'test:', con[n]
        pre = 0
        pre1 = 0
        pre2 = 0

        for k in range(4):

            train = []
            test = []
            for i in range(193):
                if i == 150:
                    continue

                ds = data[data['label'] == i]
                dtr = ds.iloc[:n]
                dte = ds.iloc[n:n + j]
                for value in dtr.values:
                    train.append(value)
                for value in dte.values:
                    test.append(value)

            train = pd.DataFrame(data=train, columns=[str(i) for i in range(4096)] + ['label'])
            test = pd.DataFrame(data=test, columns=[str(i) for i in range(4096)] + ['label'])

            X = train[[str(i) for i in range(4096)]].values
            Y = train['label'].values

            X_test = test[[str(i) for i in range(4096)]].values
            Y_test = test['label'].values

            models[0].fit(X, Y)
            models[1].fit(X, Y)
            models[2].fit(X, Y)
            #
            acc = accuracy_score(Y_test, models[0].predict(X_test))
            acc1 = accuracy_score(Y_test, models[1].predict(X_test))
            acc2 = accuracy_score(Y_test, models[2].predict(X_test))

            pre += acc
            pre1 += acc1
            pre2 += acc2
        sc1.append(pre1 / 4)
        sc2.append(pre2 / 4)
        sc.append(pre / 4)
        print 'SVM Accuracy:',pre / 4
        print 'RF Accuracy:',pre1 / 4
        print 'KNN Accuracy:',pre2 / 4
# x, y, z = [c[0] for c in sc], [c[1] for c in sc], [c[2] for c in sc]
# ax = fig.gca(projection='3d')
#
# ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
plt.xlabel('$\it{Number}$' + ' ' + '$\it{of}$' + ' ' + '$\it{training}$' + ' ' + '$\it{samples}$')
plt.ylabel('$\it{Verification}$' + ' ' + '$\it{Accuracy}$')
# from scipy.interpolate import spline
# xnew = np.linspace(1,8,300)
# power_smooth = spline(np.arange(1,8),sc,xnew)
plt.plot([1, 2, 3, 4, 5,6,7], [0.92,0.96,0.98,0.99,0.99,0.98,1], color='red', linestyle='solid',linewidth=4, antialiased=True)
#plt.plot(np.arange(1, 16), sc, color='red', linestyle='solid',linewidth=4, antialiased=True)



plt.plot([1, 2, 3, 4, 5,6,7], [0.71,0.83,0.92,0.95,0.95,0.93,0.96], color='blue', linestyle='dashdot',linewidth=4, antialiased=True)
#plt.plot(np.arange(1, 16), sc1, color='blue', linestyle='dashdot',linewidth=4, antialiased=True)




plt.plot([1, 2, 3, 4, 5,6,7], [0.31,0.84,0.91,0.96,0.96,0.97], color='green', linestyle='dotted',linewidth=4, antialiased=True)
#plt.plot(np.arange(1, 16), sc2, color='green', linestyle='dotted',linewidth=4, antialiased=True)



plt.legend(['SVM', 'RF', 'KNN'])
plt.grid()
fig.savefig('AccuracyHKPUdfilename.pdf', format='pdf')

plt.show()