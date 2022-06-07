# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import findspark
findspark.init()
from pyspark import StorageLevel
from pyspark.sql.window import Window
from pyspark.sql import SparkSession, SQLContext, Row
from pyspark.sql import functions as fun
import math, datetime
import pandas as pd
import os, sys, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py.Logger import Logger
from py.config import is_debug, ndays, k, dim
from apscheduler.schedulers.blocking import BlockingScheduler

# 主函数
def getSimilarItems(begin_date, end_date, bicf_filename):
  os.environ["PYSPARK_PYTHON"]="/usr/bin/python3.6"
  os.environ["PYSPARK_DRIVER_PYTHON"]="/usr/bin/python3.6"
  log = Logger("log/basicICF.log")

  log.logger.info("正在设置spark……")
  
  spark = SparkSession.builder \
        .master("yarn") \
        .appName("yyhBICF") \
        .config("spark.submit.deployMode","client") \
        .config("spark.port.maxRetries", "100") \
        .config("spark.sql.broadcastTimeout", "1200") \
        .config("spark.network.timeout", "600") \
        .config("spark.sql.warehouse.dir", "hdfs://mfwCluster/user/hive/warehouse") \
        .config("spark.yarn.queue", "root.search") \
        .config("spark.sql.crossJoin.enabled", "true") \
        .config("spark.executor.instances","7") \
        .config("spark.executor.memory", "10g") \
        .config("spark.driver.memory","5g") \
        .config("spark.driver.maxResultSize", "16g") \
        .enableHiveSupport() \
        .getOrCreate()       
  
  spark.sparkContext.addPyFile(os.path.abspath(__file__)) # 将所需的.py文件和依赖的库放入spark中
  # spark.stop() # 此行一定要加，否则有一天spark会内存耗尽无法启动

  log.logger.info("正在查询SQL……")
  inter_hive_sql = ''' select
    --dt,
    --search_id, --没有进行session隔离
    keyword, -- query
    CONCAT_WS("_", mapping_id, REGEXP_REPLACE(item_type,'note','travel')) as id_type, --mapping_id: 内容id或者商品id;
    REGEXP_REPLACE(item_type,'note','travel') as item_type, -- 资源类型
    duration, -- 互动页面停留时长
    interactions -- 互动事件集合
  from
    data123.table123
  where
    dt between '{}' and '{}'
    and keyword is not NULL
    and keyword <> ''
    and mapping_id is not NULL
    and duration is not NULL
    and item_type in ('weng', 'note', 'qa', 'guide')
    --and duration < 3600
  --group by
    --interactions
  --order by
    --interactions desc
    --limit 30
    '''.format(begin_date, end_date) # 如果结果要存redis，最多控制在一个月

  inter_read_df = spark.sql(inter_hive_sql)
  inter_read_df.sql_ctx.sparkSession._jsparkSession = spark._jsparkSession
  inter_read_df._sc = spark._sc
  
  # inter_read_df.show(20, False) # 查看前n行，默认查看前20行,False代表显示时太长不会被截断
  # print("总共查询到 {} 条数据".format(inter_read_df.count())) # 查看行数
  # print(len(inter_read_df.columns)) # 查看列数
  # print(inter_read_df.columns) # 查看列名

  log.logger.info("预处理：将互动数据进行量化, 并选择所使用的特征")
  click = fun.lit(1)
  vote = fun.when(fun.col("interactions").contains('vote'), 1).otherwise(0) # 正常的点赞
  comment = fun.when(fun.col("interactions").contains('comment'), 1).otherwise(0) # 一般的评论
  fav = fun.when(fun.col("interactions").contains('fav'), 1).otherwise(0) # 收藏
  share = fun.when(fun.col("interactions").contains('share'), 1).otherwise(0) # 分享
  
  inter_read_df = inter_read_df.withColumn("click", click) \
                              .withColumn("vote", vote) \
                              .withColumn("comment", comment) \
                              .withColumn("fav", fav) \
                              .withColumn("share", share) \
                              .selectExpr("keyword", "id_type", 
                                          "click * 1 as feature")
  inter_read_df.persist(StorageLevel.MEMORY_AND_DISK_SER) # 缓存起来
  # inter_read_df.show(20, False)
  
  log.logger.info("正在构建单个doc出现次数……")
  single_df = inter_read_df.groupby("id_type") \
                      .agg(fun.sum("feature").alias("feature"))
  single_df.persist(StorageLevel.MEMORY_AND_DISK_SER) # 缓存起来
  # single_df.show(20, False)
  
  # single_df.toPandas() \
          # .to_csv("./data/icfSingle.csv", index=False, header=False) # faiss的训练数据

  log.logger.info("收集query对应的doc list")
  pair_df = inter_read_df \
      .groupBy("keyword", "id_type") \
      .agg(fun.sum(fun.col("feature")).alias("feature")) \
      .withColumn("id_type_feature", fun.concat_ws(";", fun.col("id_type"), fun.col("feature"))) \
      .groupBy("keyword") \
      .agg(fun.collect_list("id_type_feature").alias("id_type_feature_list")) \
      .filter(fun.size(fun.col('id_type_feature_list')) >= 2)
  # pair_df.show(20, False)
  
  log.logger.info("正在构建doc对出现的次数")
  pair_df1 = pair_df.select("keyword",
                fun.explode(fun.col("id_type_feature_list")).alias("col1"))
  pair_df2 = pair_df1.withColumnRenamed("col1", "col2")
  
  pair_df = pair_df1.join(pair_df2,"keyword","left") \
                    .select(fun.col("col1"),fun.col("col2")) \
                    .filter(fun.col("col1") != fun.col("col2")) \
                    .select(fun.split(fun.col("col1"),";").alias("pair_feature1"),
                            fun.split(fun.col("col2"),";").alias("pair_feature2")) \
                    .select(fun.col("pair_feature1").getItem(0).alias("doc1"),
                            fun.col("pair_feature1").getItem(1).alias("feature1").cast("int"),
                            fun.col("pair_feature2").getItem(0).alias("doc2"),
                            fun.col("pair_feature2").getItem(1).alias("feature2").cast("int")) \
                    .selectExpr("doc1", "doc2", "feature1 * feature2 as feature")
  # pair_df.show(20, False)
  
  log.logger.info("正在将单个doc和doc对合并, 并计算相似度")
  single_df1 = single_df.withColumnRenamed("id_type", "doc1") \
                        .withColumnRenamed("feature", "feature1")
  single_df2 = single_df.withColumnRenamed("id_type", "doc2") \
                        .withColumnRenamed("feature", "feature2")
  pair_df = pair_df.join(single_df1, "doc1", "left") \
                  .join(single_df2, "doc2", "left") \
                  .withColumn("similar", fun.col("feature") / 
                              fun.sqrt(fun.col("feature1") * fun.col("feature2")))
  # pair_df.sort(pair_df.similar.desc()).show(20, False)
  # pair_df.printSchema()
  
  log.logger.info("正在构建相似doc序列(与每个doc最相似的top-k个doc)……")
  windowSpec  = Window.partitionBy("doc1").orderBy(fun.col("similar").desc())
  seq_df = pair_df.withColumn("group_in_number",fun.row_number().over(windowSpec))
  seq_df = seq_df.where(seq_df.group_in_number <= 15) \
                .groupBy("doc1") \
                .agg(fun.collect_list("doc2").alias("doc_list")) \
                .select(fun.col("doc1"),
                        fun.concat_ws(";",fun.col("doc_list")).alias("docs"))

  # seq_df.show(20, False)
  # seq_df.printSchema()
  
  seq_df.toPandas() \
        .to_csv(bicf_filename, index=False, header=False)

  # 清除缓存
  single_df.unpersist()
  inter_read_df.unpersist()
  spark.stop()


def timedJob():
  log = Logger("log/basicICF.log")
  log.logger.info("开始执行任务……")
  
  print("开始构建ALS embedding ……")
  bicf_i2i_path = "./data/bicf_i2i.csv"
  spark_q2i_path = "./data/spark_q2i.csv"
  bicf_q2i2i_path = './data/bicf_q2i2i.csv'
  
  begin_date = (datetime.datetime.now() +
            datetime.timedelta(days=-ndays)).strftime("%Y%m%d")
  end_date = (datetime.datetime.now() +
            datetime.timedelta(days=-1)).strftime("%Y%m%d")

  log.logger.info("step 1/3: 正在执行basicICF query找query……")
  getSimilarItems(begin_date, end_date, bicf_i2i_path)

  log.logger.info("所有任务均已执行完成。")


def gethelp():
  print("附加参数 = 0: 立即执行; 1: 定时执行")


if __name__ == "__main__":
  if is_debug:
    print("当前是debug模式, 可以放心操作")
  else:
    print("当前是线上模式, 请注意!!!")
  print(f"使用 {ndays} 天的数据进行训练")
  print(f"当前路径 {os.getcwd()}")

  if len(sys.argv) == 1:
    gethelp()
  elif len(sys.argv) == 2 and sys.argv[1] == "0":
    print("开始立即执行……")
    timedJob()
  elif len(sys.argv) == 2 and sys.argv[1] == "1":  # 定时执行
    print("开始定时任务……")
    scheduler = BlockingScheduler(timezone='Asia/Shanghai')
    scheduler.add_job(timedJob, 'cron', hour='7') # 每个n点执行
    scheduler.start()
