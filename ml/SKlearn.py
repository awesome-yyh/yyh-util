# 数据导入和切分
from sklearn import datasets
from sklearn.model_selection import train_test_split # 自动随机切分训练数据和测试数据

iris = datasets.load_iris() # 鸢尾花数据
data = iris.data
label = iris.target

# 划分训练集和测试集, random_state是随机数的种子，不填则每次都不同
X_train, X_test, y_train, y_test = train_test_split(data, label, 
                                test_size=0.2, random_state=1)



# 数据预处理
from sklearn import preprocessing

X_normalized = preprocessing.normalize(X, norm='l2') # l2正则化
X_scale = preprocessing.scale(X)  # (X-X_mean)/X_std 标准化
X_minMax = preprocessing.MinMaxScaler().fit_transform(X) # 缩放到0-1之间
binarizer = preprocessing.Binarizer(threshold=1.1) # 二值化，大于阈值为1，小于为0



# 模型训练和预测
from sklearn.linear_model import LogisticRegression as LR # 逻辑回归

lr = LR(max_iter=3000) # 模型初始化
lr.fit(train_x, train_y) # 训练
y_pred = lr.predict(test_x) # 预测测试集

print(lr) # 查看模型
print(lr.coef_) # 查看模型的最佳拟合曲线各变量的参数
print(lr.intercept_) # 查看模型的最佳拟合曲线的截距（常数项）

lr.score(test_x, test_y) # 使用逻辑回归模型自带的评分函数score获得模型在测试集上的准确性结果



# 结果分析
from sklearn import metrics # 度量均方差等

# 对于回归问题，计算均方误差，可以增减参数或更换模型等再进行训练，评价模型
print("MSE:",metrics.mean_squared_error(y_test, y_pred))

# 对于分类问题，查看正确率和混淆矩阵，斜对角线都是预测对的数
print ('ACC: %.4f' % metrics.accuracy_score(y_test, y_pred))
metrics.confusion_matrix(y_test, y_pred)
