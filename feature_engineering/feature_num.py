import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures


# filename = "data/xxx.csv"
# df_train = pd.read_csv(filename, sep='\t', encoding='utf_8_sig', header=None, index_col=None, quoting=csv.QUOTE_NONE)

df_train = pd.DataFrame({0: [1, 4, 3, 7, 6, 3]})

print("------统计特征-------")
print("最小值: ", df_train[[0]].min())
print("最大值: ", df_train[[0]].max())
print("中位数: ", df_train[[0]].median())
print("均值: ", df_train[[0]].mean())
print("方差: ", df_train[[0]].var())
print("标准差: ", df_train[[0]].std())

print("0.25分位数: ", df_train[[0]].quantile(0.25))
print("0.5分位数: ", df_train[[0]].quantile(0.5))
print("0.75分位数: ", df_train[[0]].quantile(0.75))

print("------最大最小值缩放-------")
minmax = MinMaxScaler()
age_trans = minmax.fit_transform(df_train[[0]])
print(age_trans, type(age_trans))
# df_train['Encoded_Category'] = age_trans
# print(df_train)

print("------StandardScaler(Z-score缩放)-------")
# z = (x - mean) / std
ss = StandardScaler()
age_std = ss.fit_transform(df_train[[0]])
print(age_std, type(age_std))
# df_train['Encoded_Category'] = age_std
# print(df_train)

print("------对数变换-------")
log_age = df_train[0].apply(lambda x: np.log(x))
print(pd.concat([df_train, log_age], axis=1))

print("------高次特征-------")
# （a^0，a^1, a^2, ab, b ,b^2)
ply = PolynomialFeatures(degree=2)
s = ply.fit_transform(df_train[[0]])
print(s)

print("------等距切分(分桶)-------")
print(pd.cut(df_train[0], 3, labels=['low', 'medium', 'high']))

print("------等频切分（分桶）-------")
print(pd.qcut(df_train[0], q=[0, 0.2, 0.5, 0.7, 0.8, 1]))
