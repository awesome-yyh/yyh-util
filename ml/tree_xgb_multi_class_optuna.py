import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import optuna
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split, KFold
import xgboost as xgb
from xgboost import XGBClassifier, Booster
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

# 特征工程（包括训练集和测试集）

# 划分训练集和测试集, random_state是随机数的种子，不填则每次都不同
train_df, test_df = train_test_split(df, test_size=0.2, random_state=1, shuffle=True)

# 寻找超参数试验
def objective_cv(trial):
    #Create data
    kf = KFold(n_splits=4, shuffle=True, random_state=0) # 每次都用同一个随机数打乱，然后分成4分，1分验证，其余训练
    scores = []
    for train_index, test_index in kf.split(train_df): # 拿到train和test的索引
        X_train, X_test = train_df.drop(['label'], axis=1).iloc[train_index], train_df.drop(['label'], axis=1).iloc[test_index]
        y_train, y_test = train_df['label'].iloc[train_index], train_df['label'].iloc[test_index]

        metric = objective(trial, X_train, X_test, y_train, y_test)
        scores.append(metric)
    return np.mean(scores)

def objective(trial, train_x, valid_x, train_y, valid_y, numRounds=100):
    objective_list = ['multi:softprob', 'multi:softmax']
    boosting_list = ['gbtree', 'gblinear', 'dart']
    metrics_list = ['merror', 'mlogloss']
    #Create params 
    param = {
        "verbosity": 0,
        'num_class':3,
        "objective": trial.suggest_categorical('objective', objective_list),
        "eval_metric": trial.suggest_categorical("eval_metric", metrics_list),
        "booster": trial.suggest_categorical("booster", boosting_list),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500)
    }
    #Create params 
    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    elif param["booster"] == "gblinear":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
    #Create data type XGBoost
    model = XGBClassifier(**param)  
    model.fit(train_x, train_y, eval_set=[(valid_x,valid_y)], verbose=False)
    
    preds = model.predict(valid_x)
    
    accuracy = sklearn.metrics.accuracy_score(valid_y, preds)
    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective_cv, n_trials=3)
fig =optuna.visualization.plot_param_importances(study)
fig.show()

print("xgb的最佳参数: ", study.best_trial.params) # 获取最佳参数

# 使用最佳参数训练
clf = XGBClassifier(**(study.best_trial.params))
clf.fit(train_df.drop('label', axis=1), train_df['label'], eval_set=[(test_df.drop('label', axis=1), test_df['label'])], verbose=False)

print("evals_result:", clf.evals_result())

print("显示数据特征重要度: ", clf.feature_importances_)

# 使用最佳参数预测
preds = clf.predict(test_df.drop('label', axis=1))
accuracy = sklearn.metrics.accuracy_score(test_df['label'], preds)
print("在测试集的准确率: ", accuracy)

# 模型的保存
pickle.dump(clf, open("./model/iris_xgb.pkl", "wb"))

# 模型的加载
clf = pickle.load(open("./model/iris_xgb.pkl", "rb"))

# 使用最佳参数预测
preds = clf.predict(test_df.drop('label', axis=1))
accuracy = sklearn.metrics.accuracy_score(test_df['label'], preds)
print("加载模型后，在测试集的准确率: ", accuracy)
