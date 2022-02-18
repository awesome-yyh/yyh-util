# -*- coding:utf-8 -*-
import logging
from logging.handlers import QueueHandler
import os
import re

class Logger:
    """自定义日志类
    
    typical usage example:
    
    :log = Logger("log/testLog.log")
    :log.logger.info("okkk")
    :log.logger.warning("warning")
    :log.logger.error("error")

    Args:

    :filename(str): 设置日志输出到文件的文件名,位置是相对于Python执行时的位置
    :level(str): 设置日志级别，只输出大于等于该等级的日志
    :when(str): 间隔生成新文件的时间单位，单位有：'S':秒;'M':分;'H':小时;'D':天;'midnight':每天凌晨
    :backupCount(int): 日志备份数
    """
    def __init__(self, filename, level='info', when='D', backCount=30):
        dirName = os.path.dirname(filename) # 因为日志文件会自动创建，但是文件夹得有
        if dirName != '':  # 如果给了文件夹，则创建文件夹
            os.makedirs(dirName, exist_ok=True)  # 创建时文件夹存在则跳过

        level_relations = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'crit': logging.CRITICAL
        }  # 日志级别关系映射

        self.logger = logging.getLogger(filename)  # 创建Logger实例(log记录器)
        if not self.logger.handlers:
            # 不是每次生成一个新的logger，而是先检查内存中是否存在一个叫做‘filename’的logger对象，存在则取出，不存在则新建
            # 所以如果程序中有多个同filename的logger时，日志会重复打印。故需要先判断
            self.logger.setLevel(level_relations.get(level))  # 设置日志级别

            fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
            format_str = logging.Formatter(fmt)  # 格式化器，设置日志内容的组成结构和消息字段。

            # Handlers用于将日志发送至目的地
            # 往屏幕上输出
            sh = logging.StreamHandler()  
            sh.setFormatter(format_str) 
            self.logger.addHandler(sh)

             # 往文件里写入
            th = logging.handlers.TimedRotatingFileHandler(filename=filename, interval=1, when=when, backupCount=backCount,
                                                          encoding='utf-8')
            th.setFormatter(format_str) 
            self.logger.addHandler(th)
