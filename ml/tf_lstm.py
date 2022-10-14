import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf


# 获取天气预测
# 该数据集包含14个不同的特征，例如气温，大气压力和湿度。
# 从2003年开始，每10分钟收集一次。
# 为了提高效率，本文仅使用2009年至2016年之间收集的数据。
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)
df = pd.read_csv(csv_path)
print(df.head())

# 观察温度随时间的变化
uni_data = df['T (degC)']
uni_data.index = df['Date Time']
uni_data.plot(subplots=True)
plt.show()

# 数据清洗

# 特征工程

# 划分训练集和测试集

# 搭建模型

# 训练模型

# 
