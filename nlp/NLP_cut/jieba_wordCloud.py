import jieba
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from PIL import Image
import re


# 下载中文停用词（类似于'啊'、'你'，'我'之类的词）
# > wget https://raw.githubusercontent.com/FontTian/NLP_tools/master/NLP/stopwords/stop_words_zh.txt

# 下载开源中文字体，wordcloud默认不支持线上中文
# > wget https://github.com/adobe-fonts/source-han-serif/raw/release/Variable/TTF/SourceHanSerifSC-VF.ttf


def myWord(fileName):
    print("正在读数据文件并预处理")
    text = open(fileName, encoding='utf8').read()  # 读文件
    text = re.sub(r'[^\w]', '', text)  # 只保留文字，去掉各种标点符号

    print("正在分词")
    text_cut = jieba.lcut(text)  # jieba分词，返回结果为词的列表
    text_cut = ' '.join(text_cut)  # 将分好的词用某个符号分割开连成字符串
    # print(text_cut)

    print("正在导入停用词文件")
    # 导入停词，用于去掉文本中类似于'啊'、'你'，'我'之类的词
    stop_words = open("data/stop_words_zh.txt", encoding="utf8").read().split("\n")

    print("正在生成词云")
    # 使用WordCloud生成普通词云
    word_cloud = WordCloud(width=800, height=600, 
                  font_path="./resource/SourceHanSerifSC-VF.ttf",  # 设置词云字体
                  background_color="white",  # 词云图的背景颜色
                  collocations=False,  # 让词不重复(是否包含两个单词的搭配(双字母组合))
                  stopwords=stop_words)  # 去掉的停词
    word_cloud.generate(text_cut)

    print("正在保存图片")
    word_cloud.to_file(f'wc_{fileName}.png')  # 把词云保存下来


if __name__ == "__main__":
    fileName = "./xx"
    myWord(fileName)
