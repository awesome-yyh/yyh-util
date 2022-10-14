import numpy as np
import tensorflow as tf


# 训练数据
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0

test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0

# 搭建模型
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
early_stop=tf.keras.callbacks.EarlyStopping(monitor='loss',patience=20) # 每20步检查一下是否提升
history = model.fit(training_images, training_labels, epochs=5, callbacks=[early_stop], verbose=2)

# 模型结构、预测及评估
model.summary()

classifications = model.predict(test_images)

print(f"第一张图片的预测值: {np.argmax(classifications[0])}, 概率: {np.max(classifications[0])}")
print(f"第一张图片的真实值: {test_labels[0]}")

model.evaluate(test_images, test_labels, verbose=2)
