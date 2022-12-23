import os
import re
import string
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import TFBertModel, BertTokenizer
from tf_bert import bert_encode


def convert_raw_data_into_csv():
    # Getting the names of all the raw files
    train_pos_files = os.listdir("data/aclImdb/train/pos/")
    train_neg_files = os.listdir("data/aclImdb/train/neg/")
    test_pos_files = os.listdir("data/aclImdb/test/pos/")
    test_neg_files = os.listdir("data/aclImdb/test/neg/")

    para, sentiment, datatype = ([] for i in range(3))
    for file in train_pos_files:
        with open(os.path.join("data/aclImdb/train/pos/", file)) as f:
            for line in f:
                para.append(re.sub('[\[\]/{}]+', '', line).replace("<br >",""))
                sentiment.append("pos")
                datatype.append("train")
                
    for file in train_neg_files:
        with open(os.path.join("data/aclImdb/train/neg/", file)) as f:
            for line in f:
                para.append(re.sub('[\[\]/{}]+', '', line).replace("<br >",""))
                sentiment.append("neg")
                datatype.append("train")
                
    for file in test_pos_files:
        with open(os.path.join("data/aclImdb/test/pos/", file)) as f:
            for line in f:
                para.append(re.sub('[\[\]/{}]+', '', line).replace("<br >",""))
                sentiment.append("pos")
                datatype.append("test")
                
    for file in test_neg_files:
        with open(os.path.join("data/aclImdb/test/neg/", file)) as f:
            for line in f:
                para.append(re.sub('[\[\]/{}]+', '', line).replace("<br >",""))
                sentiment.append("neg")
                datatype.append("test")

    # Saving data to a csv file named as imdb_master.csv
    df = pd.DataFrame(columns=["review", "type", "label"])
    df["review"] = para
    df["type"] = datatype
    df["label"] = sentiment

    df.to_csv(os.path.join("data","imdb_master.csv"), index=False)
    return df


def data_preprocess(data_df, max_words=10000, seq_len=100, batch_size=64):
    data_df = clean_data(data_df, ['review'])
    
    tokenizer = Tokenizer(num_words=max_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(data_df['review'].values)

    X = tokenizer.texts_to_sequences(data_df['review'].values)
    X = pad_sequences(X, maxlen=seq_len, padding='post')
    # maxlen为序列的最大长度。大于此长度的序列将被截短，小于此长度的序列将在后部填0
    # padding：'pre'或'post'，确定当需要补0时，在序列的起始还是结尾补
    y = pd.get_dummies(data_df['label']).values
    # print(type(X), type(X[0])) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
    # print(X.shape, y.shape) # (45000, 100) (45000, 2)

    # 划分训练集、验证集、测试集
    training_texts, test_texts, training_labels, test_labels = train_test_split(X,y, test_size = 0.20, random_state = 42)
    training_texts, val_texts, training_labels, val_labels = train_test_split(
        training_texts, training_labels, test_size=0.2, random_state=1, stratify=training_labels)

    # 组织tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((training_texts, training_labels))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_texts, val_labels))
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_texts, test_labels))
    test_dataset = test_dataset.batch(batch_size, drop_remainder=True)
    
    return train_dataset, val_dataset, test_dataset


def data_preprocess_bert(data_df, pretrained_model_name, from_pt=False, seq_len=100, batch_size=64):
    data_df = clean_data(data_df, ['review'])
    
    train_input_ids, train_attention_masks, token_type_ids = bert_encode(pretrained_model_name, data_df['review'].values, seq_len)
    y = pd.get_dummies(data_df['label']).values
    # print(train_input_ids.shape, train_attention_masks.shape, y.shape) # (45000, 100) (45000, 100) (45000, 2)

    # 划分训练集、验证集、测试集(bert)
    training_texts_ids, test_texts_ids = train_test_split(train_input_ids, test_size = 0.20, random_state = 42)
    training_texts_masks, test_texts_masks = train_test_split(train_attention_masks, test_size = 0.20, random_state = 42)
    training_texts_types, test_texts_types = train_test_split(token_type_ids, test_size = 0.20, random_state = 42)
    training_labels, test_labels = train_test_split(y, test_size = 0.20, random_state = 42)

    training_texts_ids, val_texts_ids = train_test_split(training_texts_ids, test_size = 0.20, random_state = 42)
    training_texts_masks, val_texts_masks = train_test_split(training_texts_masks, test_size = 0.20, random_state = 42)
    training_texts_types, val_texts_types = train_test_split(training_texts_types, test_size = 0.20, random_state = 42)
    training_labels, val_labels = train_test_split(training_labels, test_size = 0.20, random_state = 42)

    # 组织tf.data.Dataset(bert)
    train_dataset = tf.data.Dataset.from_tensor_slices((
        {"input_ids":training_texts_ids, "attention_masks": training_texts_masks, "attention_types":training_texts_types}, training_labels))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True)

    val_dataset = tf.data.Dataset.from_tensor_slices((
        {"input_ids":val_texts_ids, "attention_masks": val_texts_masks, "attention_types":val_texts_types}, val_labels))
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True)

    test_dataset = tf.data.Dataset.from_tensor_slices((
        {"input_ids":test_texts_ids, "attention_masks": test_texts_masks, "attention_types":test_texts_types}, test_labels))
    test_dataset = test_dataset.batch(batch_size, drop_remainder=True)

    return train_dataset, val_dataset, test_dataset


# clean data
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*',
          '+', '\\', '•', '~', '@', '£',
          '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',
          '½', 'à', '…', '\xa0', '\t',
          '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥',
          '▓', '—', '‹', '─', '\u3000', '\u202f',
          '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾',
          'Ã', '⋅', '‘', '∞', '«',
          '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹',
          '≤', '‡', '√', '�']

mispell_dict = {"aren't": "are not",
                "can't": "cannot",
                "couldn't": "could not",
                "couldnt": "could not",
                "didn't": "did not",
                "doesn't": "does not",
                "doesnt": "does not",
                "don't": "do not",
                "hadn't": "had not",
                "hasn't": "has not",
                "haven't": "have not",
                "havent": "have not",
                "he'd": "he would",
                "he'll": "he will",
                "he's": "he is",
                "i'd": "I would",
                "i'd": "I had",
                "i'll": "I will",
                "i'm": "I am",
                "isn't": "is not",
                "it's": "it is",
                "it'll": "it will",
                "i've": "I have",
                "let's": "let us",
                "mightn't": "might not",
                "mustn't": "must not",
                "shan't": "shall not",
                "she'd": "she would",
                "she'll": "she will",
                "she's": "she is",
                "shouldn't": "should not",
                "shouldnt": "should not",
                "that's": "that is",
                "thats": "that is",
                "there's": "there is",
                "theres": "there is",
                "they'd": "they would",
                "they'll": "they will",
                "they're": "they are",
                "theyre": "they are",
                "they've": "they have",
                "we'd": "we would",
                "we're": "we are",
                "weren't": "were not",
                "we've": "we have",
                "what'll": "what will",
                "what're": "what are",
                "what's": "what is",
                "what've": "what have",
                "where's": "where is",
                "who'd": "who would",
                "who'll": "who will",
                "who're": "who are",
                "who's": "who is",
                "who've": "who have",
                "won't": "will not",
                "wouldn't": "would not",
                "you'd": "you would",
                "you'll": "you will",
                "you're": "you are",
                "you've": "you have",
                "'re": " are",
                "wasn't": "was not",
                "we'll": " will",
                "didn't": "did not"}

puncts = puncts + list(string.punctuation)


def remove_space_links(string):
    #     string = BeautifulSoup(string).text.strip().lower()
    string = re.sub(r'((http)\S+)', 'http', string)
    string = re.sub(r'\s+', ' ', string)
    return string


def remove_numbers(x):
    x = re.sub('\d+', ' ', x)
    return x


def replace_typical_misspell(text):
    mispellings_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))

    def replace(match):
        return mispell_dict[match.group(0)]

    return mispellings_re.sub(replace, text)


# def clean_punct(x):
#     for punct in puncts:
#         x = x.replace(punct, f' {punct} ')
#     return x

def preprocessText(input_str):
    # convert text to lowercase
    input_str = input_str.lower()
    # # Remove punctuation
    # input_str = input_str.translate(str.maketrans('', '', string.punctuation))

    # repalce change lines
    input_str = input_str.replace("\r", " ")
    input_str = input_str.replace("\n", " ")
    input_str = input_str.replace("�", " ")

    #
    input_str = re.sub('[\W_]+', ' ', input_str)

    # remove white space
    output_str = re.sub(' +', ' ', input_str)
    output_str = output_str.strip()
    return output_str


def convert_str(x):
    return str(x)


def clean_data(df, cols: list):
    print("正在进行数据清洗……")
    for col in tqdm(cols):
        df[col] = df[col].apply(lambda x: convert_str(x))
        df[col] = df[col].apply(lambda x: replace_typical_misspell(x))
        df[col] = df[col].apply(lambda x: preprocessText(x))
        df[col] = df[col].apply(lambda x: remove_space_links(x))
        df[col] = df[col].apply(lambda x: remove_numbers(x))
        # df[col] = df[col].apply(lambda x: clean_punct(x))

    return df

if __name__=='__main__':
    data_df = convert_raw_data_into_csv()
    print(data_df.shape)
    print(data_df.head(10))
    print(data_df['label'].value_counts(normalize=True)) # 该列各值的占比
    y = pd.get_dummies(data_df['label']).values
    print(y[-1])
    