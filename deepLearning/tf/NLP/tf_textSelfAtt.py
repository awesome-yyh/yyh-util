import  tensorflow as tf
from tensorflow.keras.layers import Embedding, MultiHeadAttention, Dense


def point_wise_feed_forward_network(dense_size):
    ffn = tf.keras.Sequential()
    for size in dense_size:
        ffn.add(Dense(size, activation='relu'))
    return ffn

class TextSelfAtt(tf.keras.Model):

    def __init__(self,
                 maxlen,
                 max_features,
                 embedding_dims,
                 class_num,
                 last_activation='softmax',
                 dense_size=None
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
        self.dense_size = dense_size

        self.embedding = Embedding(input_dim=self.max_features, output_dim=self.embedding_dims, input_length=self.maxlen)
        self.attention = MultiHeadAttention(num_heads=4, key_dim=embedding_dims)
        
        if self.dense_size is not None:
            self.ffn = point_wise_feed_forward_network(dense_size)
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs, training=None, mask=None):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of TextBiRNNAtt must be 2, but now is {}'.format(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError('The maxlen of inputs of TextBiRNNAtt must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))

        x = self.embedding(inputs)
        x = self.attention(x, x, x)
        x = tf.reduce_mean(x, axis=1)
        # x = tf.reshape(x,  shape=(-1, 400*self.maxlen))
        if self.dense_size is not None:
            x = self.ffn(x)
        output = self.classifier(x)
        return output

    def get_config(self):
        return {"maxlen": self.maxlen,
                "max_features": self.max_features,
                "embedding_dims": self.embedding_dims,
                "class_num": self.class_num,
                "dense_size": self.dense_size}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def build_graph(self, input_shape):
        input_ = tf.keras.layers.Input(shape=input_shape)
        return tf.keras.models.Model(inputs=[input_], outputs=self.call(input_))


if __name__=='__main__':
    model = TextSelfAtt(maxlen=400,
                        max_features=5000,
                        embedding_dims=400,
                        class_num=2,
                        last_activation='softmax',
                        dense_size=[128, 64],
                        # dense_size = None
                        )
    model.build(input_shape=(None, 400))
    model.summary()
    tf.keras.utils.plot_model(model.build_graph(input_shape=400), "deepLearning/tf/NLP/text_self_att.png",
                              show_shapes=True)
