import numpy as np
import tensorflow as tf
from tensorflow.train import BytesList, FloatList, Int64List
from tensorflow.train import Feature, Features, Example


housing_median_age = tf.feature_column.numeric_column("housing_median_age")
median_house_value = tf.feature_column.numeric_column("median_house_value")

columns = [housing_median_age, median_house_value]
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


# age_mean, age_std = X_mean[1], X_std[1]  # The median age is column in 1
# housing_median_age = tf.feature_column.numeric_column(
#     "housing_median_age", normalizer_fn=lambda x: (x - age_mean) / age_std)

# median_income = tf.feature_column.numeric_column("median_income")
# bucketized_income = tf.feature_column.bucketized_column(
#     median_income, boundaries=[1.5, 3., 4.5, 6.])
# print(bucketized_income)
# ocean_prox_vocab = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
# ocean_proximity = tf.feature_column.categorical_column_with_vocabulary_list(
#     "ocean_proximity", ocean_prox_vocab)

# # Just an example, it's not used later on
# city_hash = tf.feature_column.categorical_column_with_hash_bucket(
#     "city", hash_bucket_size=1000) # = Hash(xx) % bucket_size
# city_hash

# bucketized_age = tf.feature_column.bucketized_column(
#     housing_median_age, boundaries=[-1., -0.5, 0., 0.5, 1.]) # age was scaled
# age_and_ocean_proximity = tf.feature_column.crossed_column(
#     [bucketized_age, ocean_proximity], hash_bucket_size=100)

# latitude = tf.feature_column.numeric_column("latitude")
# longitude = tf.feature_column.numeric_column("longitude")
# bucketized_latitude = tf.feature_column.bucketized_column(
#     latitude, boundaries=list(np.linspace(32., 42., 20 - 1)))
# bucketized_longitude = tf.feature_column.bucketized_column(
#     longitude, boundaries=list(np.linspace(-125., -114., 20 - 1)))
# location = tf.feature_column.crossed_column(
#     [bucketized_latitude, bucketized_longitude], hash_bucket_size=1000)

# ocean_proximity_one_hot = tf.feature_column.indicator_column(ocean_proximity)
# ocean_proximity_embed = tf.feature_column.embedding_column(ocean_proximity,
#                                                            dimension=2)
