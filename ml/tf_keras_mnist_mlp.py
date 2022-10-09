import tensorflow as tf
import numpy as np


# 训练数据
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images = training_images/255.0
test_images = test_images/255.0

# 搭建模型
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练模型
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.9):
      print("\nReached 90% accuracy so cancelling training!")
      self.model.stop_training = True # 提前终止

callbacks = myCallback()
history = model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks], verbose=2)

# 模型结构、预测及评估
model.summary()

classifications = model.predict(test_images)

print(f"第一张图片的预测值: {np.argmax(classifications[0])}, 概率: {np.max(classifications[0])}")
print(f"第一张图片的真实值: {test_labels[0]}")

model.evaluate(test_images, test_labels, verbose=2)
