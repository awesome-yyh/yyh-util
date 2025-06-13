'''
Author: yyh owyangyahe@126.com
Date: 2024-07-08 15:48:45
LastEditors: yyh owyangyahe@126.com
LastEditTime: 2024-07-08 15:48:54
FilePath: /mypython/yyh-util/reinforcement_learning/wplf.py
Description: 
'''
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
