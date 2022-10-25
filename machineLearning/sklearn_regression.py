import numpy
import sklearn.metrics
from sklearn import datasets
from sklearn.model_selection import train_test_split # 自动随机切分训练数据和测试数据

iris = datasets.load_iris() # 鸢尾花数据
data = iris.data
label = iris.target

# 划分训练集和测试集, random_state是随机数的种子，不填则每次都不同
x_train, x_test, y_train, y_test = train_test_split(data, label, 
                                test_size=0.2, random_state=1)

# # 数据处理
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

# from sklearn import preprocessing
# x_train = preprocessing.StandardScaler().fit_transform(x_train) # 标准化
# # y_train = preprocessing.StandardScaler().fit_transform(y_train) # 标准化
# x_test = preprocessing.StandardScaler().fit_transform(x_test) # 标准化
# # y_test = preprocessing.StandardScaler().fit_transform(y_test) # 标准化

# x_train = preprocessing.Normalizer().fit_transform(x_train) # 归一化
# # y_train = preprocessing.Normalizer().fit_transform(y_train) # 归一化
# x_test = preprocessing.Normalizer().fit_transform(x_test) # 归一化
# # y_test = preprocessing.Normalizer().fit_transform(y_test) # 归一化

####1.线性回归####
from sklearn.linear_model import LinearRegression, BayesianRidge, ElasticNet

model = LinearRegression()
model.fit(x_train, y_train)
pred = model.predict(x_test)
metric = numpy.sqrt(sklearn.metrics.mean_squared_error(y_test, pred))
print(f"线性回归, 均方误差: {metric}")

model = BayesianRidge()# 建立贝叶斯岭回归模型对象
model.fit(x_train, y_train)
pred = model.predict(x_test)
metric = numpy.sqrt(sklearn.metrics.mean_squared_error(y_test, pred))
print(f"贝叶斯岭回归, 均方误差: {metric}")

model = ElasticNet()  # 建立弹性网络回归模型对象
model.fit(x_train, y_train)
pred = model.predict(x_test)
metric = numpy.sqrt(sklearn.metrics.mean_squared_error(y_test, pred))
print(f"弹性网络回归, 均方误差: {metric}")

####2.SVM回归####
from sklearn import svm
model = svm.SVR(
kernel="poly", 
degree=2, # 只对kernel="poly"起作用，表示表示多项式的最高次数
gamma="scale", 
coef0=0, # 常数项
tol=0.001, # 误差项达到指定值时则停止训练
C=5, # 误差项的惩罚参数
epsilon=0.1, 
shrinking=True, 
cache_size=200, 
verbose=False, # False代表静默模式
max_iter=-1) # 默认设置为-1，表示无穷大迭代次数
model.fit(x_train, y_train)
pred = model.predict(x_test)
metric = numpy.sqrt(sklearn.metrics.mean_squared_error(y_test, pred))
print(f"SVM回归, 均方误差: {metric}")

####3.KNN回归####
from sklearn import neighbors
model = neighbors.KNeighborsRegressor()
model.fit(x_train, y_train)
pred = model.predict(x_test)
metric = numpy.sqrt(sklearn.metrics.mean_squared_error(y_test, pred))
print(f"KNN回归, 均方误差: {metric}")

####4.Bagging回归####
from sklearn.ensemble import BaggingRegressor
model = BaggingRegressor()
model.fit(x_train, y_train)
pred = model.predict(x_test)
metric = numpy.sqrt(sklearn.metrics.mean_squared_error(y_test, pred))
print(f"Bagging回归, 均方误差: {metric}")

####5.决策树回归####
from sklearn import tree
model = tree.DecisionTreeRegressor()
model.fit(x_train, y_train)
pred = model.predict(x_test)
metric = numpy.sqrt(sklearn.metrics.mean_squared_error(y_test, pred))
print(f"决策树回归, 均方误差: {metric}")

####6.随机森林回归#### Bagging + 决策树 = 随机森林
from sklearn import ensemble
model = ensemble.RandomForestRegressor(n_estimators=20)#这里使用20个决策树
model.fit(x_train, y_train)
pred = model.predict(x_test)
metric = numpy.sqrt(sklearn.metrics.mean_squared_error(y_test, pred))
print(f"随机森林回归, 均方误差: {metric}")

####7.Adaboost回归####
from sklearn import ensemble
model = ensemble.AdaBoostRegressor(n_estimators=50)#这里使用50个决策树
model.fit(x_train, y_train)
pred = model.predict(x_test)
metric = numpy.sqrt(sklearn.metrics.mean_squared_error(y_test, pred))
print(f"Adaboost回归, 均方误差: {metric}")

####8.ExtraTree极端随机树回归####
from sklearn.tree import ExtraTreeRegressor
model = ExtraTreeRegressor()
model.fit(x_train, y_train)
pred = model.predict(x_test)
metric = numpy.sqrt(sklearn.metrics.mean_squared_error(y_test, pred))
print(f"ExtraTree极端随机树回归, 均方误差: {metric}")

####9.GBRT回归####
from sklearn import ensemble
model = ensemble.GradientBoostingRegressor(
loss="squared_error", # 如果噪音点较多用"huber"，分段预测用“quantile”
learning_rate=0.02, n_estimators=200, # 学习率小时需要的迭代次数多，弱学习器的个数，或者弱学习器的最大迭代次数，太小容易欠拟合；太大容易过拟合
subsample=0.6, # 采样率，1是使用全部样本，推荐在 [0.5, 0.8] 之间
max_depth=15, 
max_features=None, # 如果样本特征数不多，比如小于50，用默认的"None"就可以了
min_samples_split=8, min_samples_leaf=1, # 如果样本量不大，不需要管这个值
verbose=0, # 0是静默，1是输出关键点，2是全部输出
criterion="friedman_mse", 
min_weight_fraction_leaf=0, 
min_impurity_decrease=0, 
init=None, random_state=None, alpha=0.9,  
max_leaf_nodes=None, warm_start=False, 
validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, 
ccp_alpha=0)
model.fit(x_train, y_train)
pred = model.predict(x_test)
metric = numpy.sqrt(sklearn.metrics.mean_squared_error(y_test, pred))
print(f"GBRT回归, 均方误差: {metric}")

