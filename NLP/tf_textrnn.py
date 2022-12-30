import  tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, GRU, Bidirectional, GlobalAveragePooling1D


class TextRNN(tf.keras.Model):

    def __init__(self,
                 maxlen,
                 max_features,
                 embedding_dims,
                 class_num,
                 last_activation='softmax'
                 ):
        '''
        :param maxlen: 文本最大长度
        :param max_features: 词典大小
        :param embedding_dims: embedding维度大小
        :param class_num: 分类数
        :param last_activation: 最后一层的激活函数
        '''
        super().__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation

        self.embedding = Embedding(input_dim=self.max_features, output_dim=self.embedding_dims, input_length=self.maxlen)
        self.bi_rnn = Bidirectional(layer=GRU(units=128, activation='tanh', return_sequences=True), merge_mode='concat' ) # LSTM or GRU
        self.avepool = GlobalAveragePooling1D()
        self.dense = Dense(128, activation='relu')
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of TextBiRNN must be 2, but now is {}'.format(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError('The maxlen of inputs of TextBiRNN must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))

        emb = self.embedding(inputs)
        x = self.bi_rnn(emb)
        x = self.avepool(x)
        x = self.dense(x)
        output = self.classifier(x)
        return output
    
    def get_config(self):
        return {"maxlen": self.maxlen,
                "max_features": self.max_features,
                "embedding_dims": self.embedding_dims,
                "class_num": self.class_num
                }
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def build_graph(self, input_shape):
        input_ = tf.keras.layers.Input(shape=input_shape)
        return tf.keras.models.Model(inputs=[input_], outputs=self.call(input_))


if __name__=='__main__':
    model = TextRNN(maxlen=400,
                    max_features=5000,
                    embedding_dims=100,
                    class_num=2,
                    last_activation='softmax',
                    )
    model.build(input_shape=(None, 400))
    model.summary()
    tf.keras.utils.plot_model(model.build_graph(input_shape=400), "NLP/text_rnn.png",
                              show_shapes=True)
