import jieba
from LAC import LAC
import thulac
import pkuseg


text = "河南台中国节气秋分篇9月23日和大家见面"

print("=== jieba ===")
print(jieba.lcut(text, cut_all=False))  # cut_all: True 完整模式, False 精确模式
print(jieba.lcut(text, cut_all=True))
print(list(jieba.cut(text, cut_all=False)))

print("=== LAC ===")
seg = LAC(mode='seg')  # 'seg'是分词模型，'lac'是分词和词性标注
print(seg.run(text))
lac = LAC(mode='lac')
print(lac.run(text))

print("=== thulac ===")
thu1 = thulac.thulac(seg_only=True)
print(thu1.cut(text, text=True))

print("=== pkuseg ===")
seg = pkuseg.pkuseg()
print(seg.cut(text))
