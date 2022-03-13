# -*- coding:utf-8 -*-
import json
import os
from hdfs3 import HDFileSystem

class Hdfs:
    def __init__(self, host, port = 9000) -> None:
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
        
        self.log.logger.info("即将从hdfs 下载："+hdfs_file)
        self.HDFS.get(hdfs_file, local_file)
        self.log.logger.info("      已经保存到："+local_file)


    def down_hdfs_dir(self, hdfs_dir, local_dir):
        # 下载文件夹中的所有hdfs文件
        hdfs_real_time_files = self.get_hdfs_list(hdfs_dir)
        for hdfs_real_time_file_abs in hdfs_real_time_files:
            self.down_hdfs_single(hdfs_real_time_file_abs, local_dir)


    def count_hdfs_tool(self, path):
      path_dir = "/xx/"
      # dir = os.listdir(path)[-1]
      # path_dir = os.path.join(path, dir)
      itemType = set()
      eventCode = set()
      num = 0
      for file in os.listdir(path_dir):
          path_file = os.path.join(path_dir, file)
          print("开始统计文件：" + path_file)
          with open(path_file, 'r') as f:
              for line in f.readlines():  # 读取所有内容并按行返回list
                  num += 1
                  json1 = json.loads(line)
                  itemType.add(json1["itemType"])
                  eventCode.add(json1["eventCode"])
      print(itemType)
      print(eventCode)
      print(num)


    def count_hdfs(self, hdfsLocalDir, as_log):  # hdfslog/xxx/, 'aslog/log.txt'
        # itemType: {'guide_article_id', 'question_id', 'travelnote_id', 'weng_id'}
        # eventCode: {'show_search', 'click_search'}
        event = dict()
        as_doc_list = count_as(as_log)
        for file in os.listdir(hdfsLocalDir):
            path_file = os.path.join(hdfsLocalDir, file)
            self.log.logger.info("开始统计文件：" + path_file)
            with open(path_file, 'r') as f:
                for line in f.readlines():  # 读取所有内容并按行返回list
                    json1 = json.loads(line)
                    doc_type = json1["itemType"].split('_', 1)[0]
                    _id = json1["itemId"] + '_' + doc_type
                    if _id in as_doc_list:  # hdfs 和 as 先join 只统计来自扶持的
                        if _id not in event.keys():  # 新的doc
                            eventInfo = dict()
                            eventInfo["doc_id"] = json1['itemId']
                            eventInfo["doc_type"] = doc_type
                            eventInfo["show_pv"] = 0
                            eventInfo["click_pv"] = 0
                            event[_id] = eventInfo

                        if json1["eventCode"] == 'show_search':
                            event[_id]["show_pv"] += 1
                        elif json1["eventCode"] == 'click_search':
                            event[_id]["click_pv"] += 1
            self.log.logger.info("该文件统计结束：" + path_file)
        # print(str(event))
        return event
