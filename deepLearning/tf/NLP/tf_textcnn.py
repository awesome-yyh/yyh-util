import  tensorflow as tf
from tensorflow.keras.layers import Embedding, Conv1D, Dense, Concatenate, GlobalMaxPooling1D


class TextCNN(tf.keras.Model):
    def __init__(self,
                 maxlen,
                 max_features,
                 embedding_dims,
                 class_num,
                 kernel_sizes=[1,2,3],
                 kernel_regularizer=None,
                 last_activation='softmax'
                 ):
        '''
        :param maxlen: 文本最大长度
        :param max_features: 词典大小
        :param embedding_dims: embedding维度大小
        :param class_num: 分类数
        :param kernel_sizes: 滑动卷积窗口大小的list, eg: [1,2,3]
        :param kernel_regularizer: eg: tf.keras.regularizers.l2(0.001)
        :param last_activation: 最后一层的激活函数
        '''
        super().__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.kernel_sizes = kernel_sizes
        
        self.embedding = Embedding(input_dim=max_features, output_dim=embedding_dims, input_length=maxlen)
        self.conv1s = []
        self.maxpools = []
        for kernel_size in kernel_sizes:
            self.conv1s.append(Conv1D(filters=128, kernel_size=kernel_size, activation='relu', kernel_regularizer=kernel_regularizer))
            self.maxpools.append(GlobalMaxPooling1D())
        # self.bn = tf.keras.layers.BatchNormalization()
        self.classifier = Dense(class_num, activation=last_activation, )

    def call(self, inputs, training=True, mask=None):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of TextCNN must be 2, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError('The maxlen of inputs of TextCNN must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))

        emb = self.embedding(inputs)
        conv1s = []
        for i in range(len(self.kernel_sizes)):
            c = self.conv1s[i](emb)
            c = self.maxpools[i](c)
            conv1s.append(c)
        x = Concatenate()(conv1s)
        # x = self.bn(x, training=training)
        output = self.classifier(x)
        return output
    
    def get_config(self):
        return {"maxlen": self.maxlen,
                "max_features": self.max_features,
                "embedding_dims": self.embedding_dims,
                "class_num": self.class_num,
                "kernel_sizes": self.kernel_sizes}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def build_graph(self, input_shape):
        input_ = tf.keras.layers.Input(shape=input_shape)
        return tf.keras.models.Model(inputs=[input_], outputs=self.call(input_))


if __name__=='__main__':
    model = TextCNN(maxlen=400,
                max_features=5000,
                embedding_dims=200,
                class_num=2,
                kernel_sizes=[2,3,5],
                kernel_regularizer=None,
                last_activation='softmax')
    model.build(input_shape=(None, 400))
    model.summary()
    tf.keras.utils.plot_model(model.build_graph(input_shape=400), "deepLearning/tf/NLP/text_cnn.png",
                              show_shapes=True)
