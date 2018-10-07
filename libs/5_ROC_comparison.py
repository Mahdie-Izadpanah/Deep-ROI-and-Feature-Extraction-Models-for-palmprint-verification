import pandas as pd
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

model = SVC(C=1.0, kernel='poly', probability=True)

train = pd.read_csv('tt2d3dnew.csv')
#train = pd.read_csv('ttIITD.csv')

X = train[[str(i) for i in range(4096)]].values
Y = np.array(train['label'].values, dtype=np.int32)

test = pd.read_csv('ts2d3dnew.csv')
#test = pd.read_csv('tsIITD.csv')

X_test = test[[str(i) for i in range(4096)]].values
Y_test = np.array(test['label'].values, dtype=np.int32)
print(Y_test)
#Y_test = label_binarize(Y_test, classes=[i for i in range(230)])

#Y = label_binarize(Y, classes=[i for i in range(230)])
#n_classes = 230

Y_test = label_binarize(Y_test, classes=[i for i in range(177)])

Y = label_binarize(Y, classes=[i for i in range(177)])
n_classes = 177

random_state = np.random.RandomState(0)
n_samples, n_features = X.shape

classifier = OneVsRestClassifier(svm.SVC(kernel='poly', probability=True,
                                         random_state=random_state))
y_score = classifier.fit(X, Y).decision_function(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], threshold = roc_curve(Y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])



lw = 2

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
print(tpr['macro'])
print(fpr['macro'])
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
fig = plt.figure()
plt.plot(fpr["macro"], tpr["macro"],
         color='red', linestyle='solid',linewidth=4)
print ("SVM auc",roc_auc["macro"])
fnr = 1 - mean_tpr
EER = all_fpr[np.nanargmin(np.absolute((fnr - all_fpr)))]
print ('SVM EER:',EER)
print (np.mean(EER))
print ('tpr:',mean_tpr)


# ________________________________________________
classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=3,))
y_score = classifier.fit(X, Y).predict_proba(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes

fpr["macro1"] = all_fpr
tpr["macro1"] = mean_tpr
roc_auc["macro1"] = auc(fpr["macro1"], tpr["macro1"])
plt.plot(fpr["macro1"], tpr["macro1"],
         color='blue', linestyle='dashdot', linewidth=4)
print(" KNN auc",roc_auc["macro1"])
fnr = 1 - mean_tpr
EER = all_fpr[np.nanargmin(np.absolute((fnr - all_fpr)))]
print ('KNN EER:',EER)
print (np.mean(EER))
print ('tpr:',mean_tpr)

# ___________________________________________________
classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=200))
y_score = classifier.fit(X, Y).predict_proba(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes

fpr["macro2"] = all_fpr
tpr["macro2"] = mean_tpr
roc_auc["macro2"] = auc(fpr["macro2"], tpr["macro2"])
plt.plot(fpr["macro2"], tpr["macro2"],
         color='green', linestyle='dotted', linewidth=4)
print ("RF auc",roc_auc["macro2"])
fnr = 1 - mean_tpr
EER = all_fpr[np.nanargmin(np.absolute((fnr - all_fpr)))]
print ('RF EER:',EER)
print (np.mean(EER))
print ('tpr:',mean_tpr)
# _________________________________________________


plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Acceptance Rate (FAR)')
plt.ylabel('Genuine Acceptance Rate (GAR)')
plt.legend(['SVM', 'KNN', 'RF'])
plt.grid()
fig.savefig('2DROCdfilename.pdf', format='pdf')
plt.show()