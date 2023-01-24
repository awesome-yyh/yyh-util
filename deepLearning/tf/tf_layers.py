import tensorflow as tf


# layers.Dense
print("------layers.Dense-------")
class CustomDense(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super(CustomDense, self).__init__()
        self.units = units

    def build(self, input_shape): # 创建层的权重, 也可在__init__中创建
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
    def get_config(self): # 使该层支持支持序列化，返回构造函数参数
        return {"units": self.units}
    
A = tf.constant([[1.0, 2.0], 
                 [3.0, 4.0]])

print('result 1:', tf.keras.layers.Dense(units=2, activation=None)(A))
print('result 2:', CustomDense(2)(A))

# layers.Dropout
print("------layers.Dropout-------")
class CustomDropout(tf.keras.layers.Layer):
    def __init__(self, rate, **kwargs):
        super(CustomDropout, self).__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        if training:
            return tf.nn.dropout(inputs, rate=self.rate)
        return inputs

A = tf.constant([[1.0, 2.0], 
                 [3.0, 4.0]])

print('result 1:', tf.keras.layers.Dropout(0.5)(A))
print('result 2:', CustomDropout(0.5)(A))

# layers.Flatten
print("------layers.Flatten-------")

print("------layers.GlobalAveragePooling1D-------")
A = tf.constant([[[1.0, 2.0], 
                 [3.0, 4.0]]])

print('result 1:', tf.keras.layers.GlobalAveragePooling1D()(A))
print('result 2:', tf.reduce_mean(A, axis=1))

# layers.Conv2D
print("------layers.Conv2D-------")

# layers.MaxPooling2D
print("------layers.MaxPooling2D-------")

# layers.SimpleRNN
print("------layers.SimpleRNN-------")
class CustomRNN(tf.keras.layers.Layer):
    def __init__(self, units):
        super(CustomRNN, self).__init__()
        self.units = units
        self.projection_1 = tf.keras.layers.Dense(units=units, activation="tanh")
        self.projection_2 = tf.keras.layers.Dense(units=units, activation="tanh")
        self.classifier = tf.keras.layers.Dense(1)

    def call(self, inputs):
        outputs = []
        state = tf.zeros(shape=(inputs.shape[0], self.units))
        for t in range(inputs.shape[1]):
            x = inputs[:, t, :]
            h = self.projection_1(x)
            y = h + self.projection_2(state)
            state = y
            outputs.append(y)
        features = tf.stack(outputs, axis=1)
        return self.classifier(features)

A = tf.constant([[[1.0, 2.0], 
                 [3.0, 4.0]]])

print('result 1: ', tf.keras.layers.SimpleRNN(2)(A))
print('result 2: ', CustomRNN(2)(A))

# layers.LSTM
print("------layers.LSTM-------")

# layers.GRU
print("------layers.GRU-------")

# layers.Bidirectional
print("------layers.Bidirectional-------")
# x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16))(inputs)
# x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(16))(inputs)

# layers.Attention
print("------layers.Attention-------")
class CustomAtt(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    def call(self, query, key_list):
        scores = tf.matmul(query, key_list, transpose_b=True)
        dk = tf.cast(tf.shape(key_list)[-1], dtype=tf.float32)
        scores /= tf.math.sqrt(dk)
        distribution = tf.nn.softmax(scores)
        return tf.matmul(distribution, key_list)
    def get_config(self):
        return {}

query = tf.constant([[1.0, 2.0]])
key_list = tf.constant([[5.0, 6.0], 
                        [7.0, 8.0]])

print('result 1:', tf.keras.layers.Attention()([query, key_list]))
print('result 2:', CustomAtt()(query, key_list))
