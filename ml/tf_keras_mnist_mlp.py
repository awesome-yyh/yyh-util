import tensorflow as tf
import numpy as np


mnist = tf.keras.datasets.mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images = training_images/255.0
test_images = test_images/255.0

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.9):
      print("\nReached 90% accuracy so cancelling training!")
      self.model.stop_training = True # 提前终止

callbacks = myCallback()

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks], verbose=2)

classifications = model.predict(test_images)

print(f"预测值: {np.argmax(classifications[0])}, 概率: {np.max(classifications[0])}")
print(f"真实值: {test_labels[0]}")

model.evaluate(test_images, test_labels, verbose=2)

model.summary()
