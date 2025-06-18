'''
Author: yyh owyangyahe@126.com
Date: 2023-09-25 10:53:49
LastEditors: yangyahe yangyahe@midu.com
LastEditTime: 2025-06-18 14:27:59
FilePath: /mypython/yyh-util/nlp/NLP_cut/pt_cut.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import jieba
from LAC import LAC
import thulac
import pkuseg


text = "河南台中国节气秋分篇9月23日和大家见面"

print("=== jieba ===")
print(jieba.lcut(text, cut_all=False))  # cut_all: True 完整模式, False 精确模式, ['河南', '台', '中国', '节气', '秋分', '篇', '9', '月', '23', '日', '和', '大家', '见面']
print(jieba.lcut(text, cut_all=True))  # ['河南', '南台', '台中', '中国', '节气', '秋分', '篇', '9', '月', '23', '日', '和', '大家', '见面']
print(list(jieba.cut(text, cut_all=False)))  # ['河南', '台', '中国', '节气', '秋分', '篇', '9', '月', '23', '日', '和', '大家', '见面']

# HMM（Hidden Markov Model）：启用HMM可以提高对未登录词的识别能力。
# use_paddle：是否使用PaddlePaddle深度学习框架的分词模型。这可能会使用GPU，但需要安装额外的依赖。
# jieba主要是CPU-based，但可以通过Paddle模式使用GPU。


print("=== LAC ===")
seg = LAC(mode='seg')  # 'seg'是分词模型，'lac'是分词和词性标注, ['河南', '台', '中国', '节气', '秋分', '篇', '9月23日', '和', '大家', '见面']
seg = LAC(mode='seg', use_cuda=False)  # LAC可以使用GPU，但需要安装paddlepaddle-gpu。
print(seg.run(text))
lac = LAC(mode='lac')  # [['河南台', '中国', '节气', '秋分篇', '9月23日', '和', '大家', '见面'], ['LOC', 'LOC', 'n', 'n', 'TIME', 'p', 'r', 'v']]
print(lac.run(text))

print("=== HanLP ===")
from pyhanlp import HanLP
print(HanLP.segment(text))

print("=== thulac ===")
thu1 = thulac.thulac(seg_only=True)
print(thu1.cut(text, text=True))

print("=== pkuseg ===")
seg = pkuseg.pkuseg()
print(seg.cut(text))
