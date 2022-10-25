import tensorflow as tf
import numpy as np


query = tf.convert_to_tensor(np.asarray([[[1., 1., 1., 3.]]]))

key_list = tf.convert_to_tensor(np.asarray([[[1., 1., 2., 4.], [4., 1., 1., 3.], [1., 1., 2., 1.]]
                                            ]))

print('query shape:', query.shape)
print('key shape:', key_list.shape)

query_value_attention_seq = tf.keras.layers.Attention()([query, key_list])

print('result 1:',query_value_attention_seq)

scores = tf.matmul(query, key_list, transpose_b=True)

distribution = tf.nn.softmax(scores)
print('distribution: ', distribution)
result = tf.matmul(distribution, key_list)
print('result 2:',query_value_attention_seq)
