import numpy as np
import pandas as pd
import tensorflow as tf


## 对数变换
log_age = df_train['Age'].apply(lambda x:np.log(x))

# MinMaxscaler（最大最小值缩放）
from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
age_trans = minmax.fit_transform(df_train[['Age']])
age_trans

# StandardScaler(Z-score缩放)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
age_std = ss.fit_transform(df_train[['Age']])
age_std

# 统计特征: 最小值、最大值、中位数、均值
df_train[['Age']].min()
df_train[['Age']].max()
df_train[['Age']].median()
df_train[['Age']].mean()
#分位数
df_train[['Age']].quantile(0.25)
df_train[['Age']].quantile(0.5)
df_train[['Age']].quantile(0.75)

# 高次特征
from sklearn.preprocessing import PolynomialFeatures
ply = PolynomialFeatures(degree = 2)
s = ply.fit_transform(df_train[['Age',"Parch"]])

# 等距切分(分桶)
df_train.loc[:,'fare_cut'] = pd.cut(df_train['Fare'],3,labels = ['low','medium','high'])
# 等频切分（分桶）
df_train.loc[:,'fare_qcut'] = pd.qcut(df_train['Fare'],q = [0,0.2,0.5,0.7,0.8,1])


# 或tfrecore解析时做特征工程
feature_columns = []

# 直接使用数值
for col in ['age','fare','parch','sibsp']:
    feature_columns.append(
        tf.feature_column.numeric_column(col))

# 数值分桶
age_buckets = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column('age'), 
    boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)
