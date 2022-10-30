import tensorflow as tf


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
