'''
Author: yyh owyangyahe@126.com
Date: 2025-03-18 17:21:10
LastEditors: yyh owyangyahe@126.com
LastEditTime: 2025-03-18 17:21:43
FilePath: /mypython/yyh-util/py/w.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np


def wilson_score(pos, total, p_z=1.96):
    """
    威尔逊得分计算函数
    参考：https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    :param pos: 正例数
    :param total: 总数
    :param p_z: 正太分布的分位数, 在95%的置信水平下，z统计量的值为1.96
    :return: 威尔逊得分
    """
    pos_rat = pos * 1. / total * 1.  # 正例比率
    score = (pos_rat + (np.square(p_z) / (2. * total))
             - ((p_z / (2. * total)) * np.sqrt(4. * total * (1. - pos_rat) * pos_rat + np.square(p_z)))) / \
            (1. + np.square(p_z) / total)
    return score


if __name__ == '__main__':
    print(wilson_score(100, 1000))