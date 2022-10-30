import numpy as np
import tensorflow as tf
from tensorflow.train import BytesList, FloatList, Int64List
from tensorflow.train import Feature, Features, Example


housing_median_age = tf.feature_column.numeric_column("housing_median_age")
median_house_value = tf.feature_column.numeric_column("median_house_value")

columns = [housing_median_age, median_house_value]
# make_parse_example_spec 用于生成这个dict，他的入参必须是个可迭代对象，一般都是一个list
feature_descriptions = tf.feature_column.make_parse_example_spec(columns)
feature_descriptions

# TFRecord读为dataset
def parse_examples(serialized_examples):
    examples = tf.io.parse_example(serialized_examples, feature_descriptions)
    targets = examples.pop("median_house_value") # separate the targets
    return examples, targets

batch_size = 32
dataset = tf.data.TFRecordDataset(["my_data_with_features.tfrecords"])
dataset = dataset.repeat().shuffle(10000).batch(batch_size).map(parse_examples)

# model.fit(dataset, epochs=5)
