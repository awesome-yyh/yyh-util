# 对文本编码
raw_documents = ["今天是星期天。",
                 "今天是星期日。",
                 "明天的明天要考试！"]

print("正在分词")
import jieba
import re # 去掉各种标点符号，只保留文字

# 将文本处理成sklearn格式: ['今天 是 星期天', '今天 是 星期日']
texts = [' '.join([word for word in jieba.cut(re.sub(r'[^\w]','', document), cut_all=False)]) for document in raw_documents]
print("sklearn texts:", texts)

# 词袋模型
print("词袋模型:")
from sklearn.feature_extraction.text import CountVectorizer
Count= CountVectorizer() # countvectorizer是一个向量化的计数器
X = Count.fit_transform(texts)
print(Count.get_feature_names()) # 分词列表
print(X.toarray()) # 句子向量

# TF-IDF
print("TF-IDF:")
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(texts)
print(tfidf.get_feature_names()) # 得到分词
print(X.toarray()) # 得到各句子的tiidf向量

# 将文本处理成gensim格式: [['今天', '是', '星期天'], ['今天', '是', '星期日']]
texts = [[word for word in jieba.cut(re.sub(r'[^\w]','', document), cut_all=False)] for document in raw_documents]
print("gensim texts:", texts)

# 词袋模型
print("词袋模型:")
from gensim import corpora, models, similarities
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
print(corpus)

# Save the Dictionary and BOW
dictionary.save('./g_dict1.dict') 
corpora.MmCorpus.serialize('./g_bow1.mm', corpus)  

# Load the Dictionary and BOW
dictionary = corpora.Dictionary.load('./g_dict1.dict')
corpus = corpora.MmCorpus('./g_bow1.mm')

print(dictionary.token2id)
print(corpus)

# TF-IDF
print("TF-IDF:")
tfidfModel = models.TfidfModel(corpus)

tfidfModel.save("./model.tfidf") # 模型保存和加载
tfidfModel = models.TfidfModel.load("./model.tfidf")

corpus_tfidf = [tfidfModel[doc] for doc in corpus]
print(corpus_tfidf)

# word2vec
print("word2vec:")
from gensim.models.word2vec import Word2Vec

w2vModel = Word2Vec(texts, min_count=1) # 先遍历一次语料库建立词典，再遍历语料库训练神经网络模型

w2vModel.save('./model.word2vec') # 模型保存和加载
w2vModel = Word2Vec.load('./model.word2vec')

print(w2vModel.wv['今天']) # 只能处理训练过的词
print(w2vModel.wv['今天'].shape)

# fasttext
print("fasttext:")
from gensim.models.fasttext import FastText

ftModel = FastText(texts, vector_size=100, window=5, min_count=1, workers=4,sg=1)
# 先遍历一次语料库建立词典，再遍历语料库训练神经网络模型
# min_count：忽略出现次数小于min_count的
# vector_size：词向量维度
# window：窗口大小,句子中当前词与目标词之间的最大距离
# alpha：初始学习率
# min_alpha：学习率线性减少到min_alpha
# sg：=1表示skip-gram(对低频词敏感),=0表示CBOW
# Hs：=1表示层次softmax，=0表示负采样

# 存储和载入模型
ftModel.save('./model.fasttext') # 保存的文件不能利用文本编辑器查看但是保存了训练的全部信息，可以在读取后追加训练
ftModel = FastText.load('./model.fasttext') # 加载模型

# 查看向量
print(ftModel.wv['今天的'])
print(ftModel.wv['今天的'].shape)

# fasttext
print("fasttext")
import fasttext

# fftModel = fasttext.load_model("./model.fasttext.bin")
fftModel = fasttext.train_unsupervised('data/tsingNews/data.txt', model='skipgram') # or 'cbow'

print(fftModel.get_sentence_vector("北京 故宫")) # 获取句子向量，单词间用空格、换行符或制表符分隔，是对词向量的平均
print(fftModel.get_dimension()) # 向量维度

# 模型的保存和加载
fftModel.save_model("./model.fasttext.bin")
fftModel = fasttext.load_model("./model.fasttext.bin")
