from tqdm import tqdm
import numpy as np
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer


def bert_encode(pretrained_model_name, data, maximum_len):
    TOKENIZER = BertTokenizer.from_pretrained(pretrained_model_name)
    input_ids = []
    attention_masks = []
    token_type_ids = []
    print("正在进行bert_encode……")
    for i in tqdm(range(len(data))):
        encoded = TOKENIZER.encode_plus(data[i],
                                        add_special_tokens=True, # Add '[CLS]' and '[SEP]'
                                        max_length=maximum_len, # # Pad & truncate all sentences
                                        padding='max_length',
                                        truncation=True,
                                        return_attention_mask=True)

        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
        token_type_ids.append(encoded['token_type_ids']) # differentiate two sentences


    return np.array(input_ids), np.array(attention_masks), np.array(token_type_ids)


def bert_model(pretrained_model_name, seq_len, class_num, from_pt=False, last_activation='softmax'):
    input_ids = tf.keras.Input(shape=(seq_len,), dtype='int32', name="input_ids")
    attention_masks = tf.keras.Input(shape=(seq_len,), dtype='int32', name="attention_masks")
    attention_types = tf.keras.Input(shape=(seq_len,), dtype='int32', name="attention_types")

    bert_model = TFBertModel.from_pretrained(pretrained_model_name, from_pt=from_pt, output_hidden_states=False)
    
    bert_output = bert_model(input_ids, attention_mask=attention_masks, token_type_ids=attention_types)
    # bert_output[0]: 各个token的向量
    # bert_output[1]: 第一个token的向量
    # bert_output[2]: 各层的输出, 需要设置output_hidden_states=True
    
    # x = tf.keras.layers.GlobalAveragePooling1D()(bert_output[0])
    x = bert_output[1]
    # x = tf.keras.layers.Concatenate(axis=2)(bert_output[2])
    # x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.15)(x)
    classifier = tf.keras.layers.Dense(class_num, activation=last_activation, name="output_1")(x)
    
    model = tf.keras.models.Model(inputs=[input_ids, attention_masks, attention_types], outputs=classifier)

    return model


if __name__=='__main__':
    # pretrained_model_name, from_pt = "hfl/chinese-roberta-wwm-ext", False
    pretrained_model_name, from_pt = "prajjwal1/bert-tiny", True
    data = ["今天是冬至，吃饺子"]
    seq_len = 100
    
    train_input_ids, train_attention_masks, token_type_ids = bert_encode(pretrained_model_name, data, seq_len)
    print(train_input_ids, train_attention_masks, token_type_ids, train_input_ids.shape)
    
    model = bert_model(pretrained_model_name, seq_len, 2, from_pt=True)
    model.summary()
    tf.keras.utils.plot_model(model, "deepLearning/tf/NLP/bert.png",
                              show_shapes=True)
    