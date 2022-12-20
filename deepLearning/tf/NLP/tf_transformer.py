import  tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Dense, Dropout, Embedding, GlobalAveragePooling1D


def point_wise_feed_forward_network(dense_size):
    ffn = tf.keras.Sequential()
    for size in dense_size:
        ffn.add(Dense(size, activation='relu'))
        ffn.add(Dropout(0.1))
    return ffn


class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.maxlen = maxlen
        self.input_dim = vocab_size
        self.output_dim = embed_dim
        
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "sequence_length": self.maxlen,
                "input_dim": self.input_dim,
            }
        )
        return config


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, dense_dim, rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [Dense(dense_dim, activation="relu"), 
             Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "dense_dim": self.dense_dim,
            }
        )
        return config


class TextTransformerEncoder(tf.keras.Model):

    def __init__(self,
                 maxlen,
                 max_features,
                 embedding_dims,
                 num_heads,
                 class_num,
                 last_activation='softmax',
                 dense_size=None,
                 ):
        '''
        :param maxlen: 文本最大长度
        :param max_features: 词典大小
        :param embedding_dims: embedding维度大小
        :param num_heads: 多头数
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
        
        self.embedding_layer = TokenAndPositionEmbedding(maxlen, max_features, embedding_dims)
        self.transformer_block = TransformerBlock(embedding_dims, num_heads, 32)
        # self.avepool = GlobalAveragePooling1D()
        if self.dense_size is not None:
            self.ffn = point_wise_feed_forward_network(dense_size)
        self.classifier = Dense(self.class_num, activation=self.last_activation)
        
    def call(self, inputs, training=None, mask=None):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of TextBiRNNAtt must be 2, but now is {}'.format(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError('The maxlen of inputs of TextBiRNNAtt must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))

        x = self.embedding_layer(inputs)
        x = self.transformer_block(x)
        # x = self.avepool()(x)
        x = tf.reduce_mean(x, axis=1)
        if self.dense_size is not None:
            x = self.ffn(x)
        output = self.classifier(x)
        return output
    
    def build_graph(self, input_shape):
        input_ = tf.keras.layers.Input(shape=input_shape)
        return tf.keras.models.Model(inputs=[input_], outputs=self.call(input_))


if __name__=='__main__':
    model = TextTransformerEncoder(maxlen=400,
                        max_features=5000,
                        embedding_dims=400,
                        num_heads=2,
                        class_num=2,
                        last_activation='softmax',
                        dense_size=[128, 64],
                        # dense_size = None
                        )
    model.build(input_shape=(None, 400))
    model.summary()
    tf.keras.utils.plot_model(model.build_graph(input_shape=400), "deepLearning/tf/NLP/text_transformer.png",
                              show_shapes=True)
