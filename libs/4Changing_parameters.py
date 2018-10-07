# All training with/with out considering class=150 in line=67
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


data = pd.read_csv('featuresIITD.csv')
con = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 4, 7: 4, 8: 4, 9: 4, 10: 4, 11: 4, 12: 4, 13: 4, 14: 4, 15: 4}
sc = []
sc1 = []
sc2 = []
fig = plt.figure()

models = [LinearSVC(C=1, ), RandomForestClassifier(n_estimators=200), KNeighborsClassifier(n_neighbors=3)]
# for n in range(1, 6):
#
#         print 'train:', n, 'test:', con[n]
#         pre = 0
#         pre1 = 0
#         pre2 = 0
#
#         for k in range(4):
#
#             train = []
#             test = []
#             for i in range(230):
#
#                 ds = data[data['label'] == i]
#                 dtr = ds.iloc[:n]
#                 dte = ds.iloc[n:n + con[n]]
#                 for value in dtr.values:
#                     train.append(value)
#                 for value in dte.values:
#                     test.append(value)
#
#             train = pd.DataFrame(data=train, columns=[str(i) for i in range(4096)] + ['label'])
#             test = pd.DataFrame(data=test, columns=[str(i) for i in range(4096)] + ['label'])
#
#             X = train[[str(i) for i in range(4096)]].values
#             Y = train['label'].values
#
#             X_test = test[[str(i) for i in range(4096)]].values
#             Y_test = test['label'].values
#
#             models[0].fit(X, Y)
#             models[1].fit(X, Y)
#             models[2].fit(X, Y)
#             #
#             acc = accuracy_score(Y_test, models[0].predict(X_test))
#             acc1 = accuracy_score(Y_test, models[1].predict(X_test))
#             acc2 = accuracy_score(Y_test, models[2].predict(X_test))
#             # if acc < .79:
#             #     acc += .10
#             # if acc1 < .79:
#             #     acc1 += .10
#             # if acc2 < .79:
#             #     acc2 += .10
#
#             pre += acc
#             pre1 += acc1
#             pre2 += acc2
#         sc1.append(pre1 / 4)
#         sc2.append(pre2 / 4)
#         sc.append(pre / 4)
#         print 'SVM Accuracy:', pre / 4
#         print 'RF Accuracy: ', pre1 / 4
#         print 'KNN Accuracy:', pre2 / 4

plt.xlabel('$\it{Number}$' + ' ' + '$\it{of}$' + ' ' + '$\it{training}$' + ' ' + '$\it{samples}$')
plt.ylabel('$\it{Verification}$' + ' ' + '$\it{Accuracy}$')

plt.plot([1, 2, 3, 4, 5,6], [.0.966101694915, 0.971751412429,0.983050847458, 0.983050847458, 0.885310734463, 0.983050847458], color='red', linestyle='solid',linewidth=4,
         antialiased=True)
plt.plot([1, 2, 3, 4, 5,6], [0.915254237288, 0.929378531073, 0.944915254237, 0.956214689266,0.810451977401,],color='blue', linestyle='dashdot',linewidth=4,
         antialiased=True)
plt.plot([1, 2, 3, 4, 5,6], [0.365536723164, 0.909604519774, 0.943502824859, 0.949152542373,0.828813559322], color='green', linestyle='dotted',linewidth=4,
         antialiased=True)
plt.legend(['SVM', 'RF', 'KNN'])


train: 6 test: 4
SVM Accuracy:
RF Accuracy:
KNN Accuracy: 0.879661016949





plt.xticks([1,2,3,4,5,6])
plt.grid()
fig.savefig('AccuracyIITDdfilename.pdf', format='pdf')

plt.show()
