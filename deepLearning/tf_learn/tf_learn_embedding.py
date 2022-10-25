import numpy as np
import tensorflow as tf


query = tf.convert_to_tensor(np.asarray([[[1., 1., 1., 3.]]]))

key_list = tf.convert_to_tensor(np.asarray([[[1., 1., 2., 4.], [4., 1., 1., 3.], [1., 1., 2., 1.]]
                                            ]))

query_value_attention_seq = tf.keras.layers.Embedding()([query, key_list])

obj_type_embedding = Embedding(len(obj_types), 3, input_length=1)(obj_type)

print('result 1:',query_value_attention_seq)