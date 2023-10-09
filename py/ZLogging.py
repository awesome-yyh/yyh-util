import os
import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler


class ZLogging:
    """多进程安全的日志类
    
    typical usage example:
    
    :zlogging = ZLogging("logs/testLog.log").logging
    :zlogging.info("okkk")
    :zlogging.warning("warning")
    :zlogging.error("error")

    Args:

    :filename(str): 设置日志输出到文件的文件名
    :level(str): 设置日志级别，只输出大于等于该等级的日志
    :maxBytesInM(int): 单个文件不超过多少M，超过时则开启新的文件进行记录
    :backupCount(int): 保留最近的多少份的日志文件，更早期的自动删除
    """
    def __init__(self, filename, level='info', maxBytesInM=19, backupCount=30):
        # 有可能路径中的文件夹不存在，故需创建（文件会自动创建，文件夹不会）
        filename = os.path.abspath(filename)
        dirpath = os.path.dirname(filename)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath)

        level_relations = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'crit': logging.CRITICAL
        }  # 日志级别关系映射

        self.logger = logging.getLogger(filename)  # 创建Logger实例(log记录器)
        # 为避免创建了多个同filename的zlogging对象时，日志会重复打印，故需先判断
        if not self.logger.handlers:
            self.logger.setLevel(level_relations.get(level))  # 设置日志级别

            format_str = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
            formatter = logging.Formatter(format_str)  # 格式化器，设置日志内容的组成结构和消息字段。

            # Handlers用于将日志发送至目的地
            # 往屏幕上输出
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            self.logger.addHandler(sh)

            # 往文件里写入
            rotateHandler = ConcurrentRotatingFileHandler(filename=filename, maxBytes=maxBytesInM * 1024 * 1024, backupCount=backupCount, encoding="utf-8")
            rotateHandler.setFormatter(formatter)
            self.logger.addHandler(rotateHandler)


if __name__ == "__main__":
    # 用法：
    zlogging = ZLogging("logs/testLog.log").logging
    zlogging.info("okkk")
    zlogging.warning("warning")
    zlogging.error("error")
    
    log_error = ZLogging("logs/testLogerr.log", level='error').logging
    log_error.error("err")
