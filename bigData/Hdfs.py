# -*- coding:utf-8 -*-
import json
import os
from hdfs3 import HDFileSystem


class HDFS:
    def __init__(self, host, port=9000) -> None:
        self.HDFS = HDFileSystem(host=host, port=port)

    def get_hdfs_list(self, hdfs_log):
        log_res = self.HDFS.ls(hdfs_log)
        return log_res

    def down_hdfs_single(self, hdfs_file, local_dir):
        # 下载单个hdfs文件
        if not os.path.exists(local_dir):
            os.makedirs(local_dir, exist_ok=True)
        
        file_name = os.path.split(hdfs_file)[1]  # 文件名
        local_file = os.path.join(local_dir, file_name)
        if os.path.exists(local_file):
            self.log.logger.info("已经有同名文件，先删除: {a}".format(a=local_file))
            os.remove(local_file)
        
        self.log.logger.info("即将从hdfs 下载：" + hdfs_file)
        self.HDFS.get(hdfs_file, local_file)
        self.log.logger.info("      已经保存到：" + local_file)

    def down_hdfs_dir(self, hdfs_dir, local_dir):
        # 下载文件夹中的所有hdfs文件
        hdfs_real_time_files = self.get_hdfs_list(hdfs_dir)
        for hdfs_real_time_file_abs in hdfs_real_time_files:
            self.down_hdfs_single(hdfs_real_time_file_abs, local_dir)
