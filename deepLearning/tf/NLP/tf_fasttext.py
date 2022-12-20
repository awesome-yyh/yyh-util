import  tensorflow as tf
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense


class FastText(tf.keras.Model):
    def __init__(self,
                maxlen,
                max_features,
                embedding_dims,
                class_num,
                last_activation = 'softmax'
                ):
        '''
        :param maxlen: 文本序列最大长度
        :param max_features: 词汇表大小
        :param embedding_dims: embedding维度大小
        :param class_num: 分类数
        :param last_activation: # 最后一层的激活函数
        '''
        super().__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        
        self.embedding = Embedding(input_dim=max_features, output_dim=embedding_dims, input_length=maxlen)
        self.pooling = GlobalAveragePooling1D()
        self.dense = Dense(128, activation='relu')
        self.classifier = Dense(self.class_num, activation=last_activation)

    def call(self, inputs, training=None, mask=None):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of FastText must be 2, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError('The maxlen of inputs of FastText must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))

        emb = self.embedding(inputs)
        pool = self.pooling(emb)
        h = self.dense(pool)
        output = self.classifier(h)
        return output

    def build_graph(self, input_shape):
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")
        _ = self.call(inputs)

    def build_graph(self, input_shape):
        input_ = tf.keras.layers.Input(shape=input_shape)
        return tf.keras.models.Model(inputs=[input_], outputs=self.call(input_))


if __name__=='__main__':
    model = FastText(maxlen=400,
                    max_features=5000,
                    embedding_dims=100,
                    class_num=2,
                    last_activation='softmax',
    )
    model.build(input_shape=(None, 400))
    model.summary()
    tf.keras.utils.plot_model(model.build_graph(400), "deepLearning/tf/NLP/fasttext.png",
                              show_shapes=True)
