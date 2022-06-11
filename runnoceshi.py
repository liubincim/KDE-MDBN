import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from scipy import interpolate as interp
from itertools import cycle

np.random.seed(1337)  # for reproducibility
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,confusion_matrix,f1_score,precision_score
from sklearn.metrics import cohen_kappa_score,hamming_loss,precision_recall_curve
from sklearn.metrics import roc_curve,auc
#分类报告
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from dbn.models import SupervisedDBNClassification
import  tensorflow
import  pandas as pd
#import eli5
#from eli5.sklearn import PermutationImportance
#import shap
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
print(tensorflow.__version__)



# Loading dataset
# digits = load_digits()
# X, Y = digits.data, digits.target
data = pd.read_csv(r'train.csv',sep=',')
#data_= pd.read_csv(r'test.csv',sep=',')
#导入数据
label_index = 8
dim  = 8
data = data.sample(frac=1.0)
data = data.reset_index()
x = data.drop(['KDE_1'],axis=1)
print(x.shape)
x = x.drop(['index'],axis=1)
x = x.drop(['x大地'],axis=1)
x = x.drop(['y大地'],axis=1)
x = np.array(x)
print(x.shape)
std=MinMaxScaler((0,1))
x=std.fit_transform(x)
y = data['KDE_1']
y = pd.to_numeric( y, errors='coerce').fillna('0').astype('int32')
# Data scaling
# X = (X / 16).astype(np.float32)


# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.3, random_state=0)#random_state=0
# Training
classifier = SupervisedDBNClassification(hidden_layers_structure=[100, 100],
                                         learning_rate_rbm=0.07, 
                                         learning_rate=0.15,
                                         n_epochs_rbm=16,
                                         n_iter_backprop=200,
                                         batch_size=32,
                                         activation_function='relu',
                                         dropout_p=0)
classifier.fit(X_train[:,2:], Y_train)

# Test
#Y的结果预测值
Y_pred = classifier.predict(X_test[:,2:])

#对应类别的概率
#二分类
#Y_pred_p = classifier.predict_proba(X_test[:,2:])[:,1]
#多分类
Y_pred_p = classifier.predict_proba_dict(X_test[:,2:])[:]

#转换成数组
Y_pred = np.array(Y_pred)
Y_test_1 = np.array(Y_test)
Y_pred_p = np.array(Y_pred_p)
Y_pred_1 = np.reshape(Y_pred,(len(Y_pred),1))
Y_test_1 = np.reshape(Y_test_1,(len(Y_test_1),1))
Y_pred_p = np.reshape(Y_pred_p,(len(Y_pred_p),1))
#Y_pred_1
print('Done.\nAccuracy: %f' % accuracy_score(Y_test, Y_pred))
#print('Done.\nprecision_score: %f' % precision_score(Y_test, Y_pred))
#print('Done.\nrecall_score: %f' % recall_score(Y_test, Y_pred))
#print('Done.\nf1_score: %f' % f1_score(Y_test, Y_pred))
print('Done.\nconfusion_matrix: ' )
print(confusion_matrix(Y_test, Y_pred))
print('Done.\ncohen_kappa_score: %f' % cohen_kappa_score(Y_test, Y_pred))
print('Done.\nhamming_loss: %f' % hamming_loss(Y_test, Y_pred))
#fpr,tpr,threshold= roc_curve(Y_test,Y_pred_p)
#print('Done.\nauc: %f' % auc(fpr, tpr))
#target_names = ['class 0','class 1','class 2', 'class 3']
print(classification_report(Y_test, Y_pred)) #,target_names=target_names

# 获取confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
#cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
 
cm = cm.astype(np.float32)
FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)
print("FP",FP)
print("FN",FN)
print("TP",TP)
print("TN",TN)
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP / (TP + FN)
# Fall out or false positive rate
FPR = FP / (FP + TN)
print("TPR",TPR)
print("FPR",FPR)

#### 新 0.3部分ROC
yy=[[0 for col in range(4)]for row in range(len(Y_pred_p))] ##获取概率
j=0
for i in Y_pred_p:
    for k in range(4):
        yy[j][k]=i[0][k]
        yy[j][k]=i[0][k]
        yy[j][k]=i[0][k]
        yy[j][k]=i[0][k]
    j=j+1
#print(yy)
#xx = [0 for col in range(531)]
#l=0
#for i in Y_pred_p:
#    xx[l]=i[0][Y_test[l]]
#    l=l+1
#print(xx)
n_class = [0,1,2,3]
y_one_hot = label_binarize(Y_test, n_class)
yy = np.array(yy)
n_classes = y_one_hot.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], thresholds = roc_curve(y_one_hot[:, i], yy[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])
print("macro：",roc_auc["macro"])

#绘图
lw=1
plt.figure()

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.3f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.3f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()

#Y的结果预测值
Y_pred = classifier.predict(X_train[:,2:])
#多分类
Y_pred_p = classifier.predict_proba_dict(X_train[:,2:])[:]
#转换成数组
Y_pred = np.array(Y_pred)
Y_test_1 = np.array(Y_train)
Y_pred_p = np.array(Y_pred_p)
Y_pred_1 = np.reshape(Y_pred,(len(Y_pred),1))
Y_test_1 = np.reshape(Y_test_1,(len(Y_test_1),1))
Y_pred_p = np.reshape(Y_pred_p,(len(Y_pred_p),1))
print('Done.\nAccuracy: %f' % accuracy_score(Y_train, Y_pred))
print('Done.\nconfusion_matrix: ' )
print(confusion_matrix(Y_train, Y_pred))
print('Done.\ncohen_kappa_score: %f' % cohen_kappa_score(Y_train, Y_pred))

cm = confusion_matrix(Y_train, Y_pred)
#cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
 
cm = cm.astype(np.float32)
FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)
print("FP",FP)
print("FN",FN)
print("TP",TP)
print("TN",TN)
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP / (TP + FN)
# Fall out or false positive rate
FPR = FP / (FP + TN)
print("TPR",TPR)
print("FPR",FPR)

#### 新 0.7部分ROC
yy=[[0 for col in range(4)]for row in range(len(Y_pred_p))] ##获取概率
j=0
for i in Y_pred_p:
    for k in range(4):
        yy[j][k]=i[0][k]
        yy[j][k]=i[0][k]
        yy[j][k]=i[0][k]
        yy[j][k]=i[0][k]
    j=j+1
n_class = [0,1,2,3]
yy = np.array(yy)
y_one_hot = label_binarize(Y_test_1, n_class)
n_classes = y_one_hot.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_one_hot[:, i].ravel(), yy[:, i].ravel())
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])
#方法2
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])
print("macro：",roc_auc["macro"])

#绘图
lw=1
plt.figure()

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.3f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.3f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
