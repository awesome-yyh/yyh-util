import tensorflow as tf


feature_columns = []

# 数值列
age = tf.feature_column.numeric_column('age')
age_buckets = tf.feature_column.bucketized_column(age, 
             boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

# 分桶列
pclass_cate = tf.feature_column.categorical_column_with_vocabulary_list(
          key='pclass', vocabulary_list=[1, 2, 3])
feature_columns.append(pclass_cate)

# 交叉列
crossed_feature = tf.feature_column.indicator_column(
    tf.feature_column.crossed_column([age_buckets, pclass_cate], hash_bucket_size=15))

feature_columns.append(crossed_feature)
