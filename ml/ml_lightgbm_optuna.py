import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import optuna
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split, KFold
import lightgbm as lgb
import pickle


# 导入数据（鸢尾花分类）
iris = sklearn.datasets.load_iris()
data = iris.data # np.array
label = iris.target # np.array
df = pd.concat([pd.DataFrame(data), pd.DataFrame(label)], axis=1, ignore_index = False)

# 数据清洗（包括训练集和测试集）
df.columns=['x1','x2','x3','x4','label'] # 花萼长度，花萼宽度，花瓣长度，花瓣宽度，鸢尾花的类别（包括Setosa，Versicolour，Virginica三类）
print(df.isna().any()) # 查看是否有缺失值

print(df.duplicated().sum()) # 统计重复的样本个数
df.drop_duplicates(inplace = True) # 重复样本删除

sns.barplot(x=df["label"].value_counts().index, 
            y=df["label"].value_counts().values, 
            palette ='coolwarm') # 检查标签的平衡性
plt.title('label')
plt.show()

# 特征工程（包括训练集和测试集）

# 划分训练集和测试集, random_state是随机数的种子，不填则每次都不同
feature, label = df.drop('label', axis=1), df['label']
X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, random_state=1, shuffle=True, stratify=label)

def objective_cv(trial):
    #Create data
    kf = KFold(n_splits=4, shuffle=True, random_state=0) # 每次都用同一个随机数打乱，然后分成4分，1分验证，其余训练
    scores = []
    for train_index, test_index in kf.split(X_train, y_train): # 拿到train和test的索引
        kf_X_train, kf_X_test = X_train.iloc[train_index], X_train.iloc[test_index]
        kf_y_train, kf_y_test = y_train.iloc[train_index], y_train.iloc[test_index]
        
        metric = objective(trial, kf_X_train, kf_X_test, kf_y_train, kf_y_test)
        scores.append(metric)
    return np.mean(scores)

def objective(trial, train_x, valid_x, train_y, valid_y, numRounds=100):
    # 参数设置
    params = {
        'verbose': -1, # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
        'task': 'train',
        'num_class': 3, # 分类数
        # 'objective': 'binary', # 二分类问题
        'objective': 'multiclass',  # 多分类问题
        # 'objective': 'regression',  # 回归问题
        'boosting_type': 'gbdt', # 设置提升类型
        'force_col_wise': 'true',
        'max_bin': 63,
        'bagging_fraction': 1.0,
        'bagging_freq': 5,
        'save_binary': True,
        'seed': 123,
        'metric': {'multi_logloss'},  # 评估函数
        'is_unbalance': True,
        'boost_from_average': False,
        "num_leaves": trial.suggest_int('num_leaves', 5, 20),
        "min_data_in_leaf": trial.suggest_int('min_data_in_leaf', 5, 20),
        "max_depth": trial.suggest_int('max_depth', 3, 15),
        "learning_rate": trial.suggest_float('learning_rate', 0.01, 0.3),
        "min_sum_hessian_in_leaf": trial.suggest_float('min_sum_hessian_in_leaf', 0.00001, 0.01),
        "feature_fraction": trial.suggest_float('feature_fraction', 0.05, 0.5),
        "lambda_l1": trial.suggest_float('lambda_l1', 0, 5.0),
        "lambda_l2": trial.suggest_float('lambda_l2', 0, 5.0),
        "min_gain_to_split": trial.suggest_float('min_gain_to_split', 0, 1.0),
        "num_boost_round": trial.suggest_int("num_boost_round", 100, 500),
    }
    
    # 转换为Dataset数据格式
    train_data = lgb.Dataset(train_x, label=train_y)
    validation_data = lgb.Dataset(valid_x, label=valid_y, reference=train_data)
    
    model = lgb.train(params, train_data,
                valid_sets=validation_data, verbose_eval = -1)
    preds = model.predict(valid_x)
    pred_y = np.argmax(preds, axis=1)
    accuracy = sklearn.metrics.accuracy_score(valid_y, pred_y)
    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective_cv, n_trials=3)
fig =optuna.visualization.plot_param_importances(study)
fig.show()

print("lgb的最佳参数: ", study.best_trial.params) # 获取最佳参数

# 使用最佳参数训练
# 转换为Dataset数据格式
train_data = lgb.Dataset(X_train, label=y_train)
validation_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

clf = lgb.train(study.best_trial.params, train_data,
                valid_sets=validation_data, verbose_eval = -1)

print('Feature names:', clf.feature_name())
print('Feature importances:', clf.feature_importance()) # 显示重要特征(值越大，说明该特征越重要)

# 使用最佳参数预测
preds = clf.predict(X_test)
pred_y = np.round(preds)
accuracy = sklearn.metrics.accuracy_score(y_test, pred_y)
print("在测试集的准确率: ", accuracy)

# 模型的保存
clf.save_model('./models/iris_lgb.txt')

# 模型的加载和预测
clf = lgb.Booster(model_file='./models/iris_lgb.txt')
preds = clf.predict(X_test)
pred_y = np.round(preds)
accuracy = sklearn.metrics.accuracy_score(y_test, pred_y)
print("加载模型后，在测试集的准确率: ", accuracy)
