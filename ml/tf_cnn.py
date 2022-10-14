import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


# 读取数据
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0

test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0

# 搭建模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3), 
              loss = tf.keras.losses.sparse_categorical_crossentropy, 
              metrics=['accuracy'])

# 训练模型
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', 
    verbose=1,
    patience=10, # 每10步检查一下是否提升
    mode='max', # monitor是准确率时max，是损失时min
    restore_best_weights=True)

history = model.fit(training_images, training_labels, 
                    validation_data = (test_images, test_labels),
                    callbacks=[early_stopping],
                    epochs=10, verbose=2)

# 模型结构、预测及评估
model.summary()

classifications = model.predict(test_images)

print(f"第一张图片的预测值: {np.argmax(classifications[0])}, 概率: {np.max(classifications[0])}")
print(f"第一张图片的真实值: {test_labels[0]}")

model.evaluate(test_images, test_labels, verbose=2)

# plotting accuracy vs validation accuracy
plt.plot(history.history['accuracy'],label = 'accuracy')
plt.plot(history.history['val_accuracy'], label='validation')
plt.legend()
plt.show()
