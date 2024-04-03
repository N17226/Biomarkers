import pandas as pd
import numpy as np
import torch
# from imblearn.ensemble import EasyEnsembleClassifier
# from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score, confusion_matrix, precision_recall_curve, auc
from sklearn.utils import shuffle

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RepeatedKFold, GridSearchCV, \
    StratifiedKFold, cross_val_predict
from scipy.stats import pearsonr, ttest_ind, levene
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import chi2, f_classif
from sklearn import svm, metrics
from sklearn.feature_selection import SelectKBest, SelectPercentile
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import sklearn
import warnings



from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        pass
        #print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_ylim(len(classes)-0.5, -0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax



psdufeature = pickle.load(open(r'.\features_1024\psdufeature.pkl', 'rb'))
psdulabel = pickle.load(open(r'.\features_1024\psdulabel.pkl', 'rb'))
validfeature = pickle.load(open(r'.\features_1024\testfeature.pkl', 'rb'))
validlabel= pickle.load(open(r'.\features_1024\testid.pkl', 'rb'))
testfeature = pickle.load(open(r'.\features_1024\validfeature.pkl', 'rb'))
testlabel= pickle.load(open(r'.\features_1024\validid.pkl', 'rb'))

num = 0
feature = np.concatenate((psdufeature, testfeature), axis=0)
label = np.concatenate((psdulabel, testlabel), axis=0)

pro=[]
labels=[]
prediction=[]
# feature = psdufeature
# label = psdulabel
label = np.expand_dims(label, axis=1)
data = np.concatenate((label, feature), axis=1)

# data = pd.concat([data_1,data_2])
# data = shuffle(data)
# data = data[np.argsort(data[:, 0])]
data = pd.DataFrame(data)
# data = data.fillna(0)
data = data.fillna(0)
list = []
for i in range(1025):
    list.append('特征{}'.format(i))
list[0] = 'label'
data.columns = list
X = data[data.columns[1:1025]]
y = data['label']
colNames = X.columns
X = X.astype(np.float64)
X = StandardScaler().fit_transform(X)
X = pd.DataFrame(X)
X.columns = colNames
########

####test
testfeature = testfeature
testlabel= testlabel
testlabel = np.expand_dims(testlabel, axis=1)
data_test = np.concatenate((testlabel, testfeature), axis=1)
data_test = pd.DataFrame(data_test)
# data = data.fillna(0)
data_test = data_test.fillna(0)
list_test = []
for ii in range(1025):
    list_test.append('特征{}'.format(ii))
list_test[0] = 'test_label'
data_test.columns = list_test
X_testall = data_test[data_test.columns[1:1025]]
y_testall = data_test['test_label']
colNames_test = X_testall.columns
X_testall = X_testall.astype(np.float64)
X_testall = StandardScaler().fit_transform(X_testall)
X_testall = pd.DataFrame(X_testall)
X_testall.columns = colNames_test
##################

########valid
validfeature = validfeature
validlabel= validlabel
validlabel = np.expand_dims(validlabel, axis=1)
data_valid = np.concatenate((validlabel, validfeature), axis=1)
data_valid = pd.DataFrame(data_valid)
# data = data.fillna(0)
data_valid = data_valid.fillna(0)
list_valid = []
for ii in range(1025):
    list_valid.append('特征{}'.format(ii))
list_valid[0] = 'valid_label'
data_valid.columns = list_valid
X_validall = data_valid[data_valid.columns[1:1025]]
y_validall = data_valid['valid_label']
colNames_valid= X_validall.columns
X_validall = X_validall.astype(np.float64)
X_validall = StandardScaler().fit_transform(X_validall)
X_validall = pd.DataFrame(X_validall)
X_validall.columns = colNames_valid
##################


train_lr = []
test_lr = []
valid_lr = []
train_svm = []
test_svm = []
valid_svm = []
train_svm_3 = []
test_svm_3 = []
valid_svm_3 = []
test_tree = []
train_tree = []
valid_tree = []
test_forest = []
valid_forest = []
train_forest = []



for i in range(100):
    print('种子：{}'.format(i))
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=8)

    #相关性分析
    X_train = X
    X_test = X_testall
    y_train = y
    y_test = y_testall
    X_valid = X_validall
    y_valid = y_validall



    # pca = PCA(n_components=50, svd_solver='full').fit(X_train)
    pca = PCA(n_components=550, svd_solver='full').fit(X_train)
    # pca = PCA(n_components=0.999).fit(X_train)
    X_train=pca.transform(X_train)
    X_test = pca.transform(X_test)
    X_valid = pca.transform(X_valid)
    total_evr = np.sum(pca.explained_variance_ratio_)
    print(X_train.shape)
    print(pca.explained_variance_ratio_.sum())


    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    X_valid= pd.DataFrame(X_valid)
    from sklearn.feature_selection import f_classif
    F, pvalues_f = f_classif(X_train, y_train)
    k = F.shape[0] - (pvalues_f > 0.05).sum()  # 所有特征减去p大于0.05的=小于0.05的特征的个数
    print(k)
    print('方差分析共筛得特征数量：{}'.format((pvalues_f < 0.05).sum()))
    # selector_new = SelectKBest(score_func=f_classif, k=18).fit(X_train, y_train)
    selector_new = SelectKBest(score_func=f_classif, k=k).fit(X_train, y_train)
    # selector_new=SelectPercentile(score_func=f_classif, percentile=80).fit(X_train,y_train)
    X_new = X_train.columns[selector_new.get_support(True)]
    X_train = X_train[X_new]
    X_test = X_test[X_new]
    X_valid = X_valid[X_new]


    # lasso检验

    alphas=np.logspace(-2.8,-1,200)
    # alphas=0.004397603609302721
    model_lassoCV=LassoCV(alphas=alphas, cv=50).fit(X_train, y_train)
    print('筛选出来的alpha系数是{}'.format(model_lassoCV.alpha_))
    coef = pd.Series(model_lassoCV.coef_, index=X_train.columns)
    print("Lasso picked"+str(sum(coef!=0))+"variables and eliminated the other"+str(sum(coef==0)))
    index = coef[coef != 0].index
    X_train = X_train[index]
    X_test = X_test[index]
    X_valid = X_valid[index]

    #



    #随机森林
    from sklearn.ensemble import RandomForestClassifier

    forest = RandomForestClassifier(n_estimators=20,class_weight='balanced',max_depth=3,min_samples_split=30,min_weight_fraction_leaf=0.29,random_state=i)

    forest.fit(X_train, y_train)
    score_test_forest = forest.score(X_test, y_test)
    score_valid_forest = forest.score(X_valid, y_valid)
    score_train_forest = forest.score(X_train, y_train)
    train_forest.append(score_train_forest)
    test_forest.append(score_test_forest)
    Precision = (y_test[y_test == forest.predict(X_test)] == 1).sum() / (forest.predict(X_test) == 1).sum()
    Npv = (y_test[y_test == forest.predict(X_test)] == 0).sum() / (forest.predict(X_test) == 0).sum()
    Recall = (y_test[y_test == forest.predict(X_test)] == 1).sum() / (y_test == 1).sum()
    Specifificity = (y_test[y_test == forest.predict(X_test)] == 0).sum() / (y_test == 0).sum()
    auc_score_forest = roc_auc_score(y_test, forest.predict_proba(X_test)[:, 1])
    ######train
    Precision3 = (y_train[y_train == forest.predict(X_train)] == 1).sum() / (forest.predict(X_train) == 1).sum()
    Npv3 = (y_train[y_train == forest.predict(X_train)] == 0).sum() / (forest.predict(X_train) == 0).sum()
    Recall3 = (y_train[y_train == forest.predict(X_train)] == 1).sum() / (y_train == 1).sum()
    Specifificity3 = (y_train[y_train == forest.predict(X_train)] == 0).sum() / (y_train == 0).sum()
    auc_score_forest3 = roc_auc_score(y_train, forest.predict_proba(X_train)[:, 1])

    print('随机森林训练集准确率是{}'.format(round(score_train_forest, 4)),
          '精确度{}、NPV{}、召回率{}、特异性{}、AUC{}'.format(round(Precision3, 4), round(Npv3, 4), round(Recall3, 4),
                                                 round(Specifificity3, 4), round(auc_score_forest3, 4)), '\n'
                                                                                                         '随机森林验证集准确率是{}'.format(
            round(score_test_forest, 4)),
          '精确度{}、Npv{}、召回率{}、特异性{}、AUC{}'.format(round(Precision, 4), round(Npv, 4), round(Recall, 4),
                                                 round(Specifificity, 4), round(auc_score_forest, 4)))
    Precision2 = (y_valid[y_valid == forest.predict(X_valid)] == 1).sum() / (forest.predict(X_valid) == 1).sum()
    Npv2 = (y_valid[y_valid == forest.predict(X_valid)] == 0).sum() / (forest.predict(X_valid) == 0).sum()
    Recall2 = (y_valid[y_valid == forest.predict(X_valid)] == 1).sum() / (y_valid == 1).sum()
    Specifificity2 = (y_valid[y_valid == forest.predict(X_valid)] == 0).sum() / (y_valid == 0).sum()
    auc_score_forest2 = roc_auc_score(y_valid, forest.predict_proba(X_valid)[:, 1])
    print('随机森林测试集准确率是{}'.format(round(score_valid_forest, 4)),
          '精确度{}、NPV{},召回率{}、特异性{}、AUC{}'.format(round(Precision2, 4), round(Npv2, 4), round(Recall2, 4),
                                                 round(Specifificity2, 4), round(auc_score_forest2, 4)))
    ttt=np.array(y_valid)
    yyyy=forest.predict_proba(X_valid)[:,1]
    precision1, recall1, thresholds1 = precision_recall_curve(ttt, forest.predict_proba(X_valid)[:,1])
    auc_precision_recall = auc(recall1, precision1)
    # print(auc_precision_recall)
    fpr, tpr, threshold = metrics.roc_curve(ttt, forest.predict_proba(X_valid)[:,1])
    roc_auc = metrics.auc(fpr, tpr)
    print(roc_auc)
    print('\n')

    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)
    y_valid = pd.Series(y_valid)
    model_LR = LogisticRegression(C=0.1, class_weight='balanced', penalty="l2", random_state=i).fit(X_train, y_train)


    score_test_lr = model_LR.score(X_test, y_test)
    score_valid_lr = model_LR.score(X_valid, y_valid)
    score_train_lr = model_LR.score(X_train, y_train)
    train_lr.append(score_train_lr)
    test_lr.append(score_test_lr)
    Precision = (y_test[y_test == model_LR.predict(X_test)] == 1).sum() / (model_LR.predict(X_test) == 1).sum()
    NPV = (y_test[y_test == model_LR.predict(X_test)] == 0).sum() / (model_LR.predict(X_test) == 0).sum()
    Recall = (y_test[y_test == model_LR.predict(X_test)] == 1).sum() / (y_test == 1).sum()
    Specifificity = (y_test[y_test == model_LR.predict(X_test)] == 0).sum() / (y_test == 0).sum()
    auc_score_LR = roc_auc_score(y_test, model_LR.predict_proba(X_test)[:, 1])

    ######################
    Precision3 = (y_train[y_train == model_LR.predict(X_train)] == 1).sum() / (model_LR.predict(X_train) == 1).sum()
    NPV3 = (y_train[y_train == model_LR.predict(X_train)] == 0).sum() / (model_LR.predict(X_train) == 0).sum()
    Recall3 = (y_train[y_train == model_LR.predict(X_train)] == 1).sum() / (y_train == 1).sum()
    Specifificity3 = (y_train[y_train == model_LR.predict(X_train)] == 0).sum() / (y_train == 0).sum()
    auc_score_forest3 = roc_auc_score(y_train, model_LR.predict_proba(X_train)[:, 1])

    print('逻辑回归训练集准确率是{}'.format(round(score_train_lr, 4)),
          '精确度{}、NPV{}、召回率{}、特异性{}、AUC{}'.format(round(Precision3, 4), round(NPV3, 4), round(Recall3, 4),
                                                 round(Specifificity3, 4), round(auc_score_forest3, 4)), '\n'
                                                                                                         '逻辑回归验证集准确率是{}'.format(
            round(score_test_lr, 4)),
          '精确度{}、NPV{}、召回率{}、特异性{}、AUC{}'.format(round(Precision, 4), round(NPV, 4), round(Recall, 4),
                                                 round(Specifificity, 4), round(auc_score_LR, 4)))
    Precision2 = (y_valid[y_valid == model_LR.predict(X_valid)] == 1).sum() / (model_LR.predict(X_valid) == 1).sum()
    NPV2 = (y_valid[y_valid == model_LR.predict(X_valid)] == 0).sum() / (model_LR.predict(X_valid) == 0).sum()
    Recall2 = (y_valid[y_valid == model_LR.predict(X_valid)] == 1).sum() / (y_valid == 1).sum()
    Specifificity2 = (y_valid[y_valid == model_LR.predict(X_valid)] == 0).sum() / (y_valid == 0).sum()
    auc_score_forest2 = roc_auc_score(y_valid, model_LR.predict_proba(X_valid)[:, 1])
    print('逻辑回归测试集准确率是{}'.format(round(score_valid_lr, 4)),
          '精确度{}、NPV{},召回率{}、特异性{}、AUC{}'.format(round(Precision2, 4), round(NPV2, 4), round(Recall2, 4),
                                                 round(Specifificity2, 4), round(auc_score_forest2, 4)))
    print('\n')
    # 决策树
    clf_tree = DecisionTreeClassifier(random_state=i, class_weight='balanced', max_depth=3, min_samples_split=10,
                                      min_weight_fraction_leaf=0.1)

    clf_tree.fit(X_train, y_train)
    y_pred = clf_tree.predict(X_test)
    y_pred_valid = clf_tree.predict(X_valid)
    score_test_tree = clf_tree.score(X_test, y_test)
    score_valid_tree = clf_tree.score(X_valid, y_valid)
    score_train_tree = clf_tree.score(X_train, y_train)
    train_tree.append(score_train_tree)
    test_tree.append(score_test_tree)
    Precision = (y_test[y_test == clf_tree.predict(X_test)] == 1).sum() / (clf_tree.predict(X_test) == 1).sum()
    Npv = (y_test[y_test == clf_tree.predict(X_test)] == 0).sum() / (clf_tree.predict(X_test) == 0).sum()
    Recall = (y_test[y_test == clf_tree.predict(X_test)] == 1).sum() / (y_test == 1).sum()
    Specifificity = (y_test[y_test == clf_tree.predict(X_test)] == 0).sum() / (y_test == 0).sum()
    auc_score_tree = roc_auc_score(y_test, clf_tree.predict_proba(X_test)[:, 1])
    ######
    Precision3 = (y_train[y_train == clf_tree.predict(X_train)] == 1).sum() / (clf_tree.predict(X_train) == 1).sum()
    Npv3 = (y_train[y_train == clf_tree.predict(X_train)] == 0).sum() / (clf_tree.predict(X_train) == 0).sum()
    Recall3 = (y_train[y_train == clf_tree.predict(X_train)] == 1).sum() / (y_train == 1).sum()
    Specifificity3 = (y_train[y_train == clf_tree.predict(X_train)] == 0).sum() / (y_train == 0).sum()
    auc_score_forest3 = roc_auc_score(y_train, clf_tree.predict_proba(X_train)[:, 1])
    print('决策树训练集准确率是{}'.format(round(score_train_tree, 4)),
          '精确度{}、NPV{}、召回率{}、特异性{}、AUC{}'.format(round(Precision3, 4), round(Npv3, 4), round(Recall3, 4),
                                                 round(Specifificity3, 4), round(auc_score_forest3, 4)), '\n'
                                                                                                         '决策树验证集准确率是{}'.format(
            round(score_test_tree, 4)),
          '精确度{}、NPV{}、召回率{}、特异性{}、AUC{}'.format(round(Precision, 4), round(Npv, 4), round(Recall, 4),
                                                 round(Specifificity, 4), round(auc_score_tree, 4)))
    Precision2 = (y_valid[y_valid == clf_tree.predict(X_valid)] == 1).sum() / (clf_tree.predict(X_valid) == 1).sum()
    Npv2 = (y_valid[y_valid == clf_tree.predict(X_valid)] == 0).sum() / (clf_tree.predict(X_valid) == 0).sum()
    Recall2 = (y_valid[y_valid == clf_tree.predict(X_valid)] == 1).sum() / (y_valid == 1).sum()
    Specifificity2 = (y_valid[y_valid == clf_tree.predict(X_valid)] == 0).sum() / (y_valid == 0).sum()
    auc_score_forest2 = roc_auc_score(y_valid, clf_tree.predict_proba(X_valid)[:, 1])
    print('决策树测试集准确率是{}'.format(round(score_valid_tree, 4)),
          '精确度{}、NPV{}、召回率{}、特异性{}、AUC{}'.format(round(Precision2, 4), round(Npv2, 4), round(Recall2, 4),
                                                 round(Specifificity2, 4), round(auc_score_forest2, 4)))
    print('\n')
    # svm
    model_svm = svm.SVC(C=0.1, random_state=i, kernel='linear', gamma='auto', probability=True,
                        class_weight='balanced').fit(X_train, y_train)


    score_test_svm = model_svm.score(X_test, y_test)
    score_valid_svm = model_svm.score(X_valid, y_valid)
    score_train_svm = model_svm.score(X_train, y_train)
    train_svm.append(score_train_svm)
    test_svm.append(score_test_svm)
    Precision = (y_test[y_test == model_svm.predict(X_test)] == 1).sum() / (model_svm.predict(X_test) == 1).sum()
    Npv = (y_test[y_test == model_svm.predict(X_test)] == 0).sum() / (model_svm.predict(X_test) == 0).sum()
    Recall = (y_test[y_test == model_svm.predict(X_test)] == 1).sum() / (y_test == 1).sum()
    Specifificity = (y_test[y_test == model_svm.predict(X_test)] == 0).sum() / (y_test == 0).sum()
    auc_score_svm = roc_auc_score(y_test, model_svm.decision_function(X_test))
    #####
    Precision3 = (y_train[y_train == model_svm.predict(X_train)] == 1).sum() / (model_svm.predict(X_train) == 1).sum()
    Npv3 = (y_train[y_train == model_svm.predict(X_train)] == 0).sum() / (model_svm.predict(X_train) == 0).sum()
    Recall3 = (y_train[y_train == model_svm.predict(X_train)] == 1).sum() / (y_train == 1).sum()
    Specifificity3 = (y_train[y_train == model_svm.predict(X_train)] == 0).sum() / (y_train == 0).sum()
    auc_score_forest3 = roc_auc_score(y_train, model_svm.decision_function(X_train))
    print('svm训练集准确率是{}'.format(round(score_train_svm, 4)),
          '精确度{}、NPV{}、召回率{}、特异性{}、AUC{}'.format(round(Precision3, 4), round(Npv3, 4), round(Recall3, 4),
                                                 round(Specifificity3, 4), round(auc_score_forest3, 4)), '\n'
                                                                                                         'svm验证集准确率是{}'.format(
            round(score_test_svm, 4)),
          '精确度{}、 NPV{}、 召回率{}、  特异性{}、    AUC{}'.format(round(Precision, 4), round(Npv, 4), round(Recall, 4),
                                                         round(Specifificity, 4), round(auc_score_svm, 4)))

    # print(model_svm.predict(X_valid))
    Precision2 = (y_valid[y_valid == model_svm.predict(X_valid)] == 1).sum() / (model_svm.predict(X_valid) == 1).sum()
    Recall2 = (y_valid[y_valid == model_svm.predict(X_valid)] == 1).sum() / (y_valid == 1).sum()
    Specifificity2 = (y_valid[y_valid == model_svm.predict(X_valid)] == 0).sum() / (y_valid == 0).sum()
    auc_score_forest2 = roc_auc_score(y_valid, model_svm.decision_function(X_valid))
    # print(model_svm.decision_function(X_valid))
    # print(model_svm.predict(X_valid))
    print('svm训练集测试集准确率是{}'.format(round(score_valid_svm, 4)),
          '精确度{}、NPV{}、召回率{}、特异性{}、AUC{}'.format(round(Precision2, 4), round(Npv2, 4), round(Recall2, 4),
                                                 round(Specifificity2, 4), round(auc_score_forest2, 4)))
    print('\n')


