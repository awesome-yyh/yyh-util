'''
Author: yyh owyangyahe@126.com
Date: 2022-11-23 21:41:47
LastEditors: yyh owyangyahe@126.com
LastEditTime: 2024-06-04 19:31:25
FilePath: /mypython/yyh-util/nlp/NLP_cut/jieba_keywords.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

from jieba import analyse



doc = '''
今年的重庆很荣幸成了抖音之星，众多抖友也开始纷纷来重庆打卡了，虽然暑假过半，但是来重庆旅游的游客依然火热，就像重庆的天气一样。为此小编特地为广大游客汇总了下重庆热门旅游景点，可作参考额！
一、抖音打卡篇
1、磁器口
重庆磁器口属国家4A级景区，位于重庆沙坪坝嘉陵江畔，蕴含丰富的巴渝文化、民间文化，是一条重庆传统街道，一直被重点保护。近年来虽然这里总被吐槽，但这依然是人山人海。
交通：轨道1号线，磁器口站下
2、解放碑
解放碑商圈里的人民解放纪念碑是重庆的标志性建筑，每年跨年时众多山城兄弟姐妹们都去这里跨年，往年众多气球零点放飞的壮丽景象就是这里。
交通：轨道2号线，临江门站下；或者轨道1号线，较场口站下
3、洪崖洞
洪崖洞由于和《千与千寻》里的场景极其相似而出名，这里刚好在长江边上，傍晚夜景美爆了，还可以乘坐两江游船观赏现实版的千与千寻。并且洪崖洞就在解放碑附近，步行即可到达。
交通：轨道1号线，小什字站下
4、长江索道
长江索道是以前的“山城空中公共汽车”，随着各座大桥的修建，以及电视节目的取景，长江索道便成为了一个旅游景点。乘坐单程索道排队长达几小时，可媲美游乐场里的排队场面了。
交通：轨道1号线，小什字站下
'''

print("基于TF/IDF")
keywords = analyse.extract_tags(doc, topK=10, withWeight=False, allowPOS=())
# topK 为返回几个 TF/IDF 权重最大的关键词，默认值为 20
# withWeight 为是否一并返回关键词权重值，默认值为 False
# allowPOS 仅包括指定词性的词，默认值为空，即不筛选, 词性表：https://blog.csdn.net/zhuzuwei/article/details/79029904
print(keywords)  # ['号线', '重庆', '索道', '解放碑', '崖洞', '磁器', '轨道', '千与千寻', '什字', '长江']

print("基于textrank")
keywords = analyse.textrank(doc, topK=10, withWeight=False)
print(keywords)  # ['重庆', '号线', '交通', '轨道', '长江', '旅游景点', '磁器', '游客', '乘坐', '汇总']
