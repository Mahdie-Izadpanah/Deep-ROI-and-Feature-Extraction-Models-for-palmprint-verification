import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC



def get_feature(vector, features):
    return [
        f for f, v in zip(features, vector) if v == 1
    ]


def all_zero(col):
    for elem in col:
        if elem != 0:
            return False
    return True


train = pd.read_csv('matrain.csv')

X = train[[str(i) for i in range(4096)]].values
Y = train['label'].values
test = pd.read_csv('matest.csv')
X_test = test[[str(i) for i in range(4096)]].values
Y_test = test['label'].values

#
model = SVC(C=1.0,kernel='poly')

model.fit(X, Y)
print('Accuracy:', accuracy_score(Y_test, model.predict(X_test)))


#
#
# data = pd.read_csv('features1.csv')
# train = []
# test = []
# for i in range(386):
#     ds = data[data['label'] == i]
#     dtr = ds.iloc[:5]
#     dte = ds.iloc[5:7]
#     for value in dtr.values:
#         train.append(value)
#     for value in dte.values:
#         test.append(value)
#
# train = pd.DataFrame(data=train, columns=[str(i) for i in range(4096)] + ['label'])
# test = pd.DataFrame(data=test, columns=[str(i) for i in range(4096)] + ['label'])
# print(train.to_csv('matrain1.csv', index=False))
# print(test.to_csv('matest1.csv', index=False))


# data = pd.read_csv('features.csv')
# sc = []
# for n in range(1, 9):
#     train = []
#     test = []
#     for i in range(1, 386, 2):
#         ds = data[data['label'] == i]
#         dtr = ds.iloc[:n]
#         dte = ds.iloc[n:n + 2]
#         for value in dtr.values:
#             train.append(value)
#         for value in dte.values:
#             test.append(value)
#     train = pd.DataFrame(data=train, columns=[str(i) for i in range(4096)] + ['label'])
#     test = pd.DataFrame(data=test, columns=[str(i) for i in range(4096)] + ['label'])
#     X = train[[str(i) for i in range(4096)]].values
#     Y = train['label'].values
#     t = pd.read_csv('matest.csv')
#     X_test = t[[str(i) for i in range(4096)]].values
#     Y_test = t['label'].values
#

# model = SVC(C=1.0, kernel='poly')
#
# model.fit(X, Y)
# sc.append(accuracy_score(Y_test, model.predict(X_test)))
#
# import matplotlib.pyplot as plt
#
# plt.plot(np.arange(1, 9), sc)
# plt.xlabel('$\it{Number}$' + ' ' + '$\it{of}$' + ' ' + '$\it{training}$' + ' ' + '$\it{samples}$')
# plt.ylabel('$\it{Verification}$' + ' ' + '$\it{Accuracy}$')
# plt.grid()
# plt.show()
#
