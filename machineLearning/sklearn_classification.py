import numpy as np
from matplotlib import pyplot as plt
import sklearn.metrics
from sklearn import datasets
from sklearn.model_selection import train_test_split # 自动随机切分训练数据和测试数据

iris = datasets.load_iris() # 鸢尾花数据
data = iris.data
label = iris.target

# 划分训练集和测试集, random_state是随机数的种子，不填则每次都不同
x_train, x_test, y_train, y_test = train_test_split(data, label, 
                                test_size=0.2, random_state=1)

print(type(x_train), x_train.shape)

# 数据处理
# import matplotlib.pyplot as plt
# plt.subplot(2, 2, 1)
# plt.plot(x_train[:, 0], y_train, 'ro', label='train')
# plt.title("train0")

# plt.subplot(2, 2, 2)
# plt.plot(x_test[:, 0], y_test, 'bo', label='test')
# plt.title("test0")

# plt.subplot(2, 2, 3)
# plt.plot(x_train[:, 1], y_train, 'ro', label='train')
# plt.title("train1")

# plt.subplot(2, 2, 4)
# plt.plot(x_test[:, 1], y_test, 'bo', label='test')
# plt.title("test1")

# plt.show()

from sklearn import preprocessing
# x_train = preprocessing.StandardScaler().fit_transform(x_train) # 标准化
# # y_train = preprocessing.StandardScaler().fit_transform(y_train) # 标准化
# x_test = preprocessing.StandardScaler().fit_transform(x_test) # 标准化
# # y_test = preprocessing.StandardScaler().fit_transform(y_test) # 标准化

# x_train = preprocessing.Normalizer().fit_transform(x_train) # 归一化
# # y_train = preprocessing.Normalizer().fit_transform(y_train) # 归一化
# x_test = preprocessing.Normalizer().fit_transform(x_test) # 归一化
# # y_test = preprocessing.Normalizer().fit_transform(y_test) # 归一化

# x_train = preprocessing.MinMaxScaler().fit_transform(x_train) # 缩放到0-1之间
# x_train = preprocessing.Binarizer(threshold=1.1).fit_transform(x_train) # 二值化，大于阈值为1，小于为0


# ####1.逻辑回归####
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(penalty='l2', max_iter=10000)
LR.fit(x_train, y_train)
pred = LR.predict(x_test)
# print(LR) # 查看模型
# print(LR.coef_) # 查看模型的最佳拟合曲线各变量的参数
# print(LR.intercept_) # 查看模型的最佳拟合曲线的截距（常数项）
print ('LR ACC: %.4f' % sklearn.metrics.accuracy_score(y_test, pred))

####2.SVM分类####
from sklearn import svm
SVM = svm.SVC(
kernel="poly", 
degree=2, # 只对kernel="poly"起作用，表示表示多项式的最高次数
gamma="scale", 
coef0=0, # 常数项
tol=0.001, # 误差项达到指定值时则停止训练
C=5, # 误差项的惩罚参数
shrinking=True, 
cache_size=200, 
verbose=False, # False代表静默模式
max_iter=-1) # 默认设置为-1，表示无穷大迭代次数
SVM.fit(x_train, y_train)
pred = SVM.predict(x_test)
print ('SMV ACC: %.4f' % sklearn.metrics.accuracy_score(y_test, pred))


####3.KNN分类####
from sklearn import neighbors
KNN = neighbors.KNeighborsClassifier()
KNN.fit(x_train, y_train)
pred = KNN.predict(x_test)
print ('KNN ACC: %.4f' % sklearn.metrics.accuracy_score(y_test, pred))


####4.朴素贝叶斯分类####
from sklearn import naive_bayes
NaiveBayes = naive_bayes.GaussianNB()
NaiveBayes.fit(x_train, y_train)
pred = NaiveBayes.predict(x_test)
print ('朴素贝叶斯分类 ACC: %.4f' % sklearn.metrics.accuracy_score(y_test, pred))


####5.Bagging分类####
from sklearn.ensemble import BaggingClassifier
Bagging = BaggingClassifier(naive_bayes.GaussianNB())
Bagging.fit(x_train, y_train)
pred = Bagging.predict(x_test)
print ('Bagging分类 ACC: %.4f' % sklearn.metrics.accuracy_score(y_test, pred))


####6.决策树分类####
from sklearn import tree
DecisionTree = tree.DecisionTreeClassifier()
DecisionTree.fit(x_train, y_train)
pred = DecisionTree.predict(x_test)
print ('决策树分类 ACC: %.4f' % sklearn.metrics.accuracy_score(y_test, pred))

# # 决策树可视化(文字)
# print(tree.export_text(DecisionTree))
# # 决策树可视化(图片)
# _ = tree.plot_tree(
#     DecisionTree, 
#     feature_names=iris.feature_names,  
#     class_names=iris.target_names,
#     filled=True
# )
# plt.show()


####7.随机森林分类####
from sklearn import ensemble
RandomForest = ensemble.RandomForestClassifier(n_estimators=20)#这里使用20个决策树
RandomForest.fit(x_train, y_train)
pred = RandomForest.predict(x_test)
print ('随机森林分类 ACC: %.4f' % sklearn.metrics.accuracy_score(y_test, pred))

# 随机森林中第一棵决策树可视化
from sklearn import tree
# print(len(RandomForest.estimators_), type(RandomForest.estimators_))
tree.export_graphviz(RandomForest.estimators_[0],
                out_file='machineLearning/fig/RandomForest.dot', 
                node_ids=True,
                feature_names = iris.feature_names,
                class_names = iris.target_names,
                rounded = True, proportion = False, 
                precision = 2, filled = True)
from subprocess import call
call(['dot', '-Tpng', 'machineLearning/fig/RandomForest.dot', '-o', 'machineLearning/fig/RandomForest.png', '-Gdpi=600'])


####8.Adaboost分类####
from sklearn import ensemble
Adaboost = ensemble.AdaBoostClassifier(n_estimators=50)#这里使用50个决策树
Adaboost.fit(x_train, y_train)
pred = Adaboost.predict(x_test)
print ('Adaboost分类 ACC: %.4f' % sklearn.metrics.accuracy_score(y_test, pred))

# Adaboost可视化
from sklearn import tree
# print(len(Adaboost.estimators_), type(Adaboost.estimators_))
tree.export_graphviz(Adaboost.estimators_[0],
                out_file='machineLearning/fig/Adaboost.dot', 
                node_ids=True,
                feature_names = iris.feature_names,
                class_names = iris.target_names,
                rounded = True, proportion = False, 
                precision = 2, filled = True)
from subprocess import call
call(['dot', '-Tpng', 'machineLearning/fig/Adaboost.dot', '-o', 'machineLearning/fig/Adaboost.png', '-Gdpi=600'])


####9.GBDT分类####
from sklearn import ensemble
GBDT = ensemble.GradientBoostingClassifier(
loss="log_loss", # 如果噪音点较多用"huber"，分段预测用“quantile”
learning_rate=0.02, 
n_estimators=200, # 弱学习器的个数
subsample=0.6, # 采样率，1是使用全部样本，推荐在 [0.5, 0.8] 之间
max_depth=15, 
max_features=None, # 如果样本特征数不多，比如小于50，用默认的"None"就可以了
min_samples_split=8, min_samples_leaf=1, # 如果样本量不大，不需要管这个值
verbose=0, # 0是静默，1是输出关键点，2是全部输出
criterion="friedman_mse", 
min_weight_fraction_leaf=0, 
min_impurity_decrease=0, 
init=None, random_state=None,
max_leaf_nodes=None, warm_start=False, 
validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, 
ccp_alpha=0)

GBDT.fit(x_train, y_train)
pred = GBDT.predict(x_test)
print ('GBDT分类 ACC: %.4f' % sklearn.metrics.accuracy_score(y_test, pred))
print(f"GBDT特征重要度: {GBDT.feature_importances_}")

# GBDT可视化
from sklearn import tree
# print(len(GBDT.estimators_), type(GBDT.estimators_), GBDT.estimators_.shape)
tree.export_graphviz(GBDT.estimators_[5, 0], # 绘制训练6棵树
                out_file='machineLearning/fig/GBDT.dot', 
                node_ids=True,
                feature_names = iris.feature_names,
                class_names = iris.target_names,
                rounded = True, proportion = False, 
                precision = 2, filled = True)
from subprocess import call
call(['dot', '-Tpng', 'machineLearning/fig/GBDT.dot', '-o', 'machineLearning/fig/GBDT.png', '-Gdpi=600'])


####10.GBDT+LR分类####
# apply返回训练数据X_train在训练好的模型里每棵树中所处的叶子节点的位置
gbdtLeafTrain = GBDT.apply(x_train)
# print(gbdtLeafTrain[0][0])
gbdtLeafTrain = gbdtLeafTrain.reshape(-1, gbdtLeafTrain.shape[1]*gbdtLeafTrain.shape[2])
gbdtLeafTest = GBDT.apply(x_test)
gbdtLeafTest = gbdtLeafTest.reshape(-1, gbdtLeafTest.shape[1]*gbdtLeafTest.shape[2])
# print(gbdtLeafTrain.shape)

# # 进行One-hot 操作
# from sklearn.preprocessing import  OneHotEncoder
# enc = OneHotEncoder()
# enc.fit(gbdtLeafTrain)
# enc.fit(gbdtLeafTest)
# gbdtLeafTrain = np.array(enc.transform(gbdtLeafTrain).toarray())
# gbdtLeafTest = np.array(enc.transform(gbdtLeafTest).toarray())
# print(gbdtLeafTrain.shape)

# 使用新的特征进行逻辑回归
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(penalty='l2', max_iter=10000)
LR.fit(gbdtLeafTrain, y_train)
pred = LR.predict(gbdtLeafTest)
print ('GBDT+LR ACC: %.4f' % sklearn.metrics.accuracy_score(y_test, pred))
