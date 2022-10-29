import datetime
import os, pathlib, shutil, random
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf


# 读取数据
# !curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# !tar -xf aclImdb_v1.tar.gz
# !rm -r aclImdb/train/unsup

base_dir = pathlib.Path("data/aclImdb")
val_dir = base_dir / "val"
train_dir = base_dir / "train"
for category in ("neg", "pos"):
    os.makedirs(val_dir / category)
    files = os.listdir(train_dir / category)
    random.Random(1337).shuffle(files)
    num_val_samples = int(0.2 * len(files))
    val_files = files[-num_val_samples:]
    for fname in val_files:
        shutil.move(train_dir / category / fname,
                    val_dir / category / fname)

# 数据探索分析

# 数据清洗(缺失值、重复值、异常值、大小写、标点)

# 数据采样(搜集、合成、过采样、欠采样、阈值移动、loss加权、评价指标)

# 特征工程(数值、文本、类别、时间)
# raw_data = preprocessing.StandardScaler().fit_transform(raw_data)

# 划分训练集、验证集、测试集
batch_size = 32
train_ds = tf.keras.utils.text_dataset_from_directory(
    "data/aclImdb/train", batch_size=batch_size
)
val_ds = tf.keras.utils.text_dataset_from_directory(
    "data/aclImdb/val", batch_size=batch_size
)
test_ds = tf.keras.utils.text_dataset_from_directory(
    "data/aclImdb/test", batch_size=batch_size
)

text_only_train_ds = train_ds.map(lambda x, y: x)

max_length = 600
max_tokens = 20000
text_vectorization = tf.keras.layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_length,
)
text_vectorization.adapt(text_only_train_ds)

int_train_ds = train_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)
int_val_ds = val_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)
int_test_ds = test_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)

# 搭建模型
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = tf.keras.layers.Embedding(
            input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = tf.keras.layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
        })
        return config

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = tf.keras.Sequential(
            [tf.keras.layers.Dense(dense_dim, activation="relu"),
             tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(
            inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config

sequence_length = 600
vocab_size = 20000
embed_dim = 256
dense_dim = 32
num_heads = 2

inputs = tf.keras.Input(shape=(None,), dtype="int64")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(inputs)
x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
x = tf.keras.layers.GlobalMaxPooling1D()(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)

# 查看模型结构
# model.build((None,))
model.summary()
tf.keras.utils.plot_model(model, "model.png", show_shapes=True)

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3), 
              loss = "mse", metrics=["mae"])

# 训练模型
# early_stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_mae',
    verbose=1,
    patience=10, # 每10步检查一下是否提升
    mode='max', # monitor是准确率时max，是损失时min
    restore_best_weights=True)

# tensorboard
log_dir="./logs/transformer/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1, write_graph=True,
    write_images=True, write_steps_per_second=False, update_freq='epoch',
    profile_batch=0, embeddings_freq=0, embeddings_metadata=None)

# 保存ckpt文件
ckpt_file_path = "./models/transformerckpt/"
ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
    ckpt_file_path, verbose=0, 
    monitor='val_loss', mode='min',
    save_freq='epoch', save_best_only=False,
    save_weights_only=True, 
    options=None, initial_value_threshold=None
)

# 断点续训
# if os.path.exists(ckpt_file_path):
#     model.load_weights(ckpt_file_path)
#     # 若成功加载前面保存的参数，输出下列信息
#     print("checkpoint_loaded")

history = model.fit(int_train_ds, 
                    validation_data = int_val_ds,
                    callbacks=[early_stopping, tensorboard_callback, ckpt_callback],
                    batch_size = 128, epochs=3, verbose=2)

# 模型评估和改进
# > tensorboard --logdir logs/mlp
import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.subplot(121)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(122)
acc = history_dict["mae"]
val_acc = history_dict["val_mae"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

test_loss, test_mae = model.evaluate(int_test_ds, verbose=2)
print("在测试集的准确率: ", test_mae)

# 模型保存加载和部署
# 模型加载和预测(ckpt)
if os.path.exists(ckpt_file_path):
    model.load_weights(ckpt_file_path)
    classifications = model.predict(int_test_ds)
    print(f"第一个预测值: {np.argmax(classifications[0])}, 概率: {np.max(classifications[0])}")

# 模型的保存(pb)
pb_file_path = './models/multiModel/transformer/1'
tf.keras.models.save_model(model, pb_file_path)

# 模型加载和预测(pb)
restored_saved_model=tf.keras.models.load_model(pb_file_path)
test_input = np.expand_dims(list(int_test_ds.unbatch())[0][0],0)
pred = restored_saved_model.predict(test_input) # 模型预测
print(f"第一个预测值: {np.argmax(pred)}")

# restored_saved_model.get_layer("dense_1").kernel # 查看模型参数

# 最后使用docker-tf-serving部署pb文件的模型，即可使用http在线访问预测
