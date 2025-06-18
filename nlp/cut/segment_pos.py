'''
Author: yangyahe yangyahe@midu.com
Date: 2025-06-17 14:35:39
LastEditors: yangyahe yangyahe@midu.com
LastEditTime: 2025-06-18 14:24:48
FilePath: /app/yangyahe/proofcheck/third_party/zdiff/src/zdiff/segment_pos.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import jieba
import jieba.posseg as pseg


class SegmentPOS:
    def __init__(self, mode="lac"):
        jieba.initialize()
        self.mode = mode
        
    def run(self, text: str):
        if self.mode == "lac":
            words = pseg.lcut(text)
            words_all, flags_all = [], []
            for word, flag in words:
                words_all.append(word)
                flags_all.append(flag)
            return words_all, flags_all
        else:
            return jieba.lcut(text, cut_all=False)

    def add_word(self, word, sep=None, tag=None):
        """添加单词
        """
        if sep:
            for item in word.strip().split(sep):
                jieba.add_word(item, freq=None, tag=tag)
        else:
            jieba.add_word(word, freq=None, tag=tag)
    

POS = SegmentPOS(mode="lac")
Segment = SegmentPOS(mode='seg')

for i in ['一', '二', '三', '四', '五', '六', '七', '八', '九']:
    prefix = f"第{i}完全小学"
    Segment.add_word(prefix)
Segment.add_word('文综 理综', sep=' ')
Segment.add_word('城德印象')


if __name__ == "__main__":
    print(POS.run("我在第一完全小学上学"))
    POS.add_word('第一完全小学上学')
    print(POS.run("我在第一完全小学上学"))
    
    print(POS.run("这是二氧化碳"))
    