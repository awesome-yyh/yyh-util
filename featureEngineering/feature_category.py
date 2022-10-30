import pandas as pd
import tensorflow as tf


# get_dummies: pandas的one-hot encoding
# 当特征为字符串形式的类别型特征时，比如“Embarked”代表登船口岸
embarked_oht = pd.get_dummies(df_train[['Embarked']])
# 当特征为字符串形式的数值型特征时，比如“Pclass”代表船舱等级，其取值为[1,2,3],用数字代表不同等级的船舱，本质上还是类别型特征
Pclass_oht = pd.get_dummies(df_train['Pclass'].apply(lambda x:str(x)))


# 或tfrecore解析时做特征工程
feature_columns = []

# 类别列
# 注意：所有的Catogorical Column类型最终都要通过indicator_column/embedding_column转换成Dense Column类型才能传入模型！！
# indicator_column 是一个onehot工具，用于把sparse特征进行onehot 变换

# categorical_column_with_vocabulary_list 只能用于数量较少的种类，比如性别，省份等。
sex = tf.feature_column.indicator_column(
      tf.feature_column.categorical_column_with_vocabulary_list(
      key='sex',vocabulary_list=["male", "female"]))
feature_columns.append(sex)

pclass = tf.feature_column.indicator_column(
      tf.feature_column.categorical_column_with_vocabulary_list(
      key='pclass',vocabulary_list=[1,2,3]))
feature_columns.append(pclass)

embarked = tf.feature_column.indicator_column(
      tf.feature_column.categorical_column_with_vocabulary_list(
      key='embarked',vocabulary_list=['S','C','B']))
feature_columns.append(embarked)


# categorical_column_with_identity 用于已经事先编码的sparse特征，例如，店铺id虽然数量非常大，但是已经把每个店铺id都从0开始编码
poi = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_identity("poi", num_buckets=10, default_value=0))
feature_columns.append(poi)


# categorical_column_with_hash_bucket 如果sparse特征非常庞大，hash_bucket_size的大小应该留有充分的冗余量，否则非常容易出现hash冲突，在这个例子中，一共有3个店铺，把hash_bucket_size设定为10，仍然得到了hash冲突的结果，这样poi的信息就被丢失了一些信息
ticket = tf.feature_column.indicator_column(
     tf.feature_column.categorical_column_with_hash_bucket('ticket',3))
feature_columns.append(ticket)


# 嵌入列 embedding_column 用于生成embedding后的张量
cabin = tf.feature_column.embedding_column(
    tf.feature_column.categorical_column_with_hash_bucket('cabin',32),dimension=2)
feature_columns.append(cabin)
