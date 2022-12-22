from tqdm import tqdm
import numpy as np
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer


def bert_encode(TOKENIZER, data, maximum_len):
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


def bert_model(pretrained_model_name, maxlen, class_num, from_pt=False, last_activation='softmax'):

    input_ids = tf.keras.Input(shape=(maxlen,), dtype='int32', name="input_ids")
    attention_masks = tf.keras.Input(shape=(maxlen,), dtype='int32', name="attention_masks")
    attention_types = tf.keras.Input(shape=(maxlen,), dtype='int32', name="attention_types")

    bert_base = TFBertModel.from_pretrained(pretrained_model_name, from_pt=from_pt)
    transformer_layer = bert_base([input_ids, attention_masks, attention_types])

    output = transformer_layer[1]

    # output = tf.keras.layers.Dense(128, activation='relu')(output)
    # output = tf.keras.layers.Dropout(0.2)(output)

    output = tf.keras.layers.Dense(class_num, activation=last_activation)(output)

    model = tf.keras.models.Model(inputs=[input_ids, attention_masks, attention_types], outputs=output)

    return model


if __name__=='__main__':
    # pretrained_model_name, from_pt = "hfl/chinese-roberta-wwm-ext", False
    pretrained_model_name, from_pt = "prajjwal1/bert-tiny", True
    
    data = ["今天是冬至，吃饺子"]
    seq_len = 100
    
    TOKENIZER = BertTokenizer.from_pretrained(pretrained_model_name)
    train_input_ids, train_attention_masks, token_type_ids = bert_encode(TOKENIZER, data, seq_len)
    print(train_input_ids, train_attention_masks, token_type_ids, train_input_ids.shape)
    
    model = bert_model(pretrained_model_name, seq_len, 2, from_pt=True)
    model.summary()
    tf.keras.utils.plot_model(model, "deepLearning/tf/NLP/bert.png",
                              show_shapes=True)
    