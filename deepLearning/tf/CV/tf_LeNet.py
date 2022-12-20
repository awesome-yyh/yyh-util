import tensorflow as tf
from tensorflow import keras


class LeNet_5(keras.Model):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.c1 = keras.layers.Conv2D(filters=6, kernel_size=(5,5), padding='valid', activation="tanh")
        self.s2 = keras.layers.MaxPooling2D(pool_size=(2,2))
        self.c3 = keras.layers.Conv2D(filters=16, kernel_size=(5,5), padding='valid', activation="tanh")
        self.s4 = keras.layers.MaxPooling2D(pool_size=(2,2))
        self.flatten = keras.layers.Flatten()
        self.f5 = keras.layers.Dense(120, activation="tanh")
        self.f6 = keras.layers.Dense(84, activation="tanh")
        self.f7 = keras.layers.Dense(num_classes, activation="softmax")

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.s2(x)
        x = self.c3(x)
        x = self.s4(x)
        x = self.flatten(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        
        return x
    def build_graph(self, input_shape):
        input_ = tf.keras.layers.Input(shape=input_shape)
        return tf.keras.models.Model(inputs=[input_], outputs=self.call(input_))


if __name__ == '__main__':
    model = LeNet_5(num_classes=10)
    model.build(input_shape=(None, 28, 28, 1))
    model.summary()
    tf.keras.utils.plot_model(model.build_graph(input_shape=(28, 28, 1)), "deepLearning/tf/CV/lenet.png",
                              show_shapes=True)
