import lightgbm as lgb

# 参数设置
params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    # 'objective': 'regression',  # 回归问题
    'objective': 'multiclass',  # 多分类问题
    'num_class': 3, # 分类树
    'metric': {'l2', 'auc'},  # 评估函数
    'max_depth': 4, # 树的最大深度，过拟合时可以减小此值
    'learning_rate': 0.1, # 学习率
    'lambda_l1': 0.1, # 正则化参数
    'lambda_l2': 0.2,
    'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
}

# 数据预处理
train_data = lgb.Dataset(X_train, label=y_train) # 训练集
validation_data = lgb.Dataset(X_test, label=y_test, reference=train_data) # 验证集

# 训练和评估
# 模型训练
gbm = lgb.train(params, train_data, num_boost_round=200, 
                valid_sets=validation_data, early_stopping_rounds=1)

# 模型保存
gbm.save_model('model.txt')

# 模型加载
gbm = lgb.Booster(model_file='model.txt')

print('Feature names:', gbm.feature_name())
print('Feature importances:', gbm.feature_importance()) # 显示重要特征(值越大，说明该特征越重要)


# 模型预测
y_pred = gbm.predict(X_test)
y_pred = [list(x).index(max(x)) for x in y_pred] # 对于分类问题需要
