import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Dense
from transformers import shape_list, BertTokenizer, TFBertModel


def convert_examples_to_features(
            examples,
            labels,
            max_seq_len,
            tokenizer,
            pad_token_id_for_segment=0,
            pad_token_id_for_label=-100):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id

    input_ids, attention_masks, token_type_ids, data_labels = [], [], [], []
    tag_to_index = {tag: index for index, tag in enumerate(labels)}

    for example, label in tqdm(zip(examples, labels), total=len(examples)):
        tokens = []
        labels_ids = []
        for one_word, label_token in zip(example, label):
            subword_tokens = tokenizer.tokenize(one_word)
            tokens.extend(subword_tokens)
            labels_ids.extend(
                [tag_to_index[label_token]] + [pad_token_id_for_label] * (len(subword_tokens) - 1)
            )

        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[: (max_seq_len - special_tokens_count)]
            labels_ids = labels_ids[: (max_seq_len - special_tokens_count)]

        tokens += [sep_token]
        labels_ids += [pad_token_id_for_label]

        tokens = [cls_token] + tokens
        labels_ids = [pad_token_id_for_label] + labels_ids

        input_id = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_id)
        padding_count = max_seq_len - len(input_id)
        input_id = input_id + ([pad_token_id] * padding_count)
        attention_mask = attention_mask + ([0] * padding_count)
        token_type_id = [pad_token_id_for_segment] * max_seq_len
        label = labels_ids + ([pad_token_id_for_label] * padding_count)

        assert len(input_id) == max_seq_len, "Error with input length {} vs {}".format(
            len(input_id), max_seq_len
        )
        assert (
            len(attention_mask) == max_seq_len
        ), "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_id) == max_seq_len, "Error with token type length {} vs {}".format(
            len(token_type_id), max_seq_len
        )
        assert len(label) == max_seq_len, "Error with labels length {} vs {}".format(
            len(label), max_seq_len
        )

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        data_labels.append(label)

    input_ids = np.array(input_ids, dtype=int)
    attention_masks = np.array(attention_masks, dtype=int)
    token_type_ids = np.array(token_type_ids, dtype=int)
    data_labels = np.asarray(data_labels, dtype=np.int32)

    return (input_ids, attention_masks, token_type_ids), data_labels


class TFBertForTokenClassification(tf.keras.Model):
    def __init__(self, model_name, class_num, last_activation = 'softmax'):
        super().__init__()
        self.class_num = class_num
        
        self.bert = TFBertModel.from_pretrained(model_name, from_pt=True)
        self.classifier = Dense(self.class_num, activation=last_activation)

    def call(self, inputs):
        input_ids, attention_mask, token_type_ids = inputs
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        all_output = outputs[0]
        prediction = self.classifier(all_output)

        return prediction
    def build_graph(self, input_shape):
        input_ = tf.keras.layers.Input(shape=input_shape)
        return tf.keras.models.Model(inputs=[input_], outputs=self.call(input_))


if __name__=='__main__':
    # tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
    # X, y = convert_examples_to_features(
    #     train_data_sentence, train_data_label, max_seq_len=128, tokenizer=tokenizer
    # )
    pretrained_model_name = "hfl/chinese-roberta-wwm-ext"
    model = TFBertForTokenClassification(model_name=pretrained_model_name,
                    class_num=2,
                    last_activation='softmax',
    )
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    inputs = tokenizer.encode_plus("你好", max_length=100, padding='max_length')
    ipt = []
    ipt.append(inputs['input_ids'])
    ipt.append(inputs['input_ids'])
    ipt = np.asarray(ipt, dtype=np.int32)

    attn = []
    attn.append(inputs['attention_mask'])
    attn.append(inputs['attention_mask'])
    attn = np.asarray(attn, dtype=np.int32)

    ids = []
    ids.append(inputs['token_type_ids'])
    ids.append(inputs['token_type_ids'])
    ids = np.asarray(ids, dtype=np.int32)
    data = [ipt, attn, ids]
    model.build(input_shape=[ipt.shape, attn.shape, ids.shape])
    model.summary()
    # tf.keras.utils.plot_model(model.build_graph(400), "deepLearning/tf/NLP/bert.png",
                            #   show_shapes=True)
