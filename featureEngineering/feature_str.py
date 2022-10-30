import tokenization


# 对文本编码

# 词袋模型
#countvectorizer是一个向量化的计数器
from sklearn.feature_extraction.text import CountVectorizer
vec= CountVectorizer()
doc = {
    '今天 是 晴天',
    '今天 星期天',
    '星期天 今天'
}
X = vec.fit_transform(doc)
print(vec.get_feature_names()) # 分词列表
print(X.toarray()) # 句子向量

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(doc)
X.toarray() #得到tiidf的值
tfidf.get_feature_names()#得到特征值

# bert(token)
str = "大哥大"
max_seq_length = 12
tokenizer = tokenization.FullTokenizer(vocab_file="vocab.txt")

tokens = tokenizer.tokenize(str)
print(tokens)

# Account for [CLS] and [SEP] with "- 2"
if len(tokens) > max_seq_length - 2:
    tokens = tokens[:max_seq_length - 2]

strtoken = ["[CLS]"]
strtoken.extend(tokens)
strtoken.append("[SEP]")
print(strtoken, type(strtoken))

input_ids = tokenizer.convert_tokens_to_ids(strtoken)

while len(input_ids) < max_seq_length:
    input_ids.append(0)

print(input_ids, type(input_ids), len(input_ids))
