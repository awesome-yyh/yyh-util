import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


print(f"tf version: {tf.__version__}")

# y = 2*x - 1
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape = [1]))

model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(xs, ys, epochs=300)

model.summary() # 查看模型结构

print(model.predict([10.0]))

# plt.plot(xs, ys, 'ro', label='Original data')
# plt.show()
