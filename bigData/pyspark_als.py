import pandas as pd
import findspark
from pyspark import StorageLevel
from pyspark.sql.window import Window
from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as fun
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import IndexToString, StringIndexer, Normalizer
from pyspark.ml.linalg import Vectors
import datetime
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py.ZLogging import ZLogging


def get_als_embedding(begin_date, end_date, query_filename, doc_filename):
    """得到2个embedding文件, 以及id到query及embedding的映射

    Args:
        begin_date (str): 起始日期, 如: 20220313
        end_date (str): 结束日期, 如: 20220313

    Returns:
        tuple: query_embedding_map, doc_embedding_map, id_query_map, id_doc_map
        ps. 同时也会得到2个文件: ./data/query_embedding.csv, ./data/doc_embedding.csv
    """
    
    findspark.init()
    os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.6"
    os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/bin/python3.6"
    log = ZLogging("log/als.log")

    log.logger.info("正在设置spark……")

    # master 的地址，提交任务到哪里执行
    # appName 是在yarn管理界面查看的应用名称
    # deployMode 在本地 (client) 启动 driver 或在 cluster 上启动，默认是 client
    # spark.network.timeout 单位是秒，默认120秒
    # executor.instances executor个数
    # executor.memory  每个 executor 的内存，默认是1G, 不要用动态分配，容易崩溃
    # driver.memory  Driver内存，默认 1G
    spark = SparkSession.builder \
        .master("yarn") \
        .appName("yyhALS") \
        .config("spark.submit.deployMode", "client") \
        .config("spark.port.maxRetries", "100") \
        .config("spark.sql.broadcastTimeout", "1200") \
        .config("spark.network.timeout", "600") \
        .config("spark.sql.warehouse.dir", "hdfs://mfwCluster/user/hive/warehouse") \
        .config("spark.yarn.queue", "root.search") \
        .config("spark.executor.instances", "7") \
        .config("spark.executor.memory", "10g") \
        .config("spark.driver.memory", "5g") \
        .config("spark.driver.maxResultSize", "16g") \
        .enableHiveSupport() \
        .getOrCreate()
    
    spark.sparkContext.addPyFile(os.path.abspath(__file__))  # 将所需的.py文件和依赖的库放入spark中
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
      '''.format(begin_date, end_date)  # 如果结果要存redis，最多控制在一个月

    inter_read_df = spark.sql(inter_hive_sql)
    inter_read_df.sql_ctx.sparkSession._jsparkSession = spark._jsparkSession
    inter_read_df._sc = spark._sc
    
    click = fun.lit(1)
    vote = fun.when(fun.col("interactions").contains('vote'), 1).otherwise(0)  # 点赞
    comment = fun.when(fun.col("interactions").contains('comment'), 1).otherwise(0)  # 评论
    fav = fun.when(fun.col("interactions").contains('fav'), 1).otherwise(0)  # 收藏
    share = fun.when(fun.col("interactions").contains('share'), 1).otherwise(0)  # 分享
    
    inter_read_df = inter_read_df.withColumn("click", click) \
                                .withColumn("vote", vote) \
                                .withColumn("comment", comment) \
                                .withColumn("fav", fav) \
                                .withColumn("share", share)
    # inter_read_df.show(30, False) # 查看前n行，默认查看前20行,False代表显示时太长不会被截断
    # inter_read_df.printSchema()
    # log.logger.info(f"总共查询到 {inter_read_df.count()} 条数据") # 查看行数

    log.logger.info("根据频次生成index列, index从0开始")
    queryIndexModel = StringIndexer(inputCol="keyword", outputCol="queryIndex").fit(inter_read_df)
    queryIndex_df = queryIndexModel.transform(inter_read_df)
    queryIndex_df.sql_ctx.sparkSession._jsparkSession = spark._jsparkSession
    queryIndex_df._sc = spark._sc

    docIndexModel = StringIndexer(inputCol="id_type", outputCol="docIndex").fit(inter_read_df)
    index_df = docIndexModel.transform(queryIndex_df)
    index_df.sql_ctx.sparkSession._jsparkSession = spark._jsparkSession
    index_df._sc = spark._sc

    # 合并
    index_df = index_df.groupBy(["keyword", "queryIndex", "id_type", "item_type", "docIndex"]).agg(
        fun.sum("duration").alias("dur_sum"),
        fun.sum("click").alias("click_sum"),
        fun.sum("vote").alias("vote_sum"),
        fun.sum("comment").alias("comment_sum"),
        fun.sum("fav").alias("fav_sum"),
        fun.sum("share").alias("share_sum"))
    
    index_df.persist(StorageLevel.MEMORY_AND_DISK_SER)  # 缓存起来
    # log.logger.info("按总时长排序: ")
    # index_df.orderBy(fun.col("click_sum").desc()).show(30, False)

    log.logger.info("ALS训练数据: ")
    index_df.show(20, False)

    log.logger.info("开始训练ALS……")
    alsModel = ALS(maxIter=20,   # 最大迭代次数，可以调节，但超过40会程序崩溃(和regParam以及训练数据量相互制约)
                   rank=128,  # 向量的维数
                   checkpointInterval=10,  # 每训练几步缓存一次
                   regParam=0.003,  # 正则化参数
                   alpha=1,  # 置信参数
                   implicitPrefs=True,  # 隐式评分
                   userCol="queryIndex",  # user列
                   itemCol="docIndex",  # item列
                   ratingCol="click_sum",  # 评分列
                   coldStartStrategy="drop"  # 设置冷启动策略，以确保我们没有获得NaN评估指标
                   ).fit(index_df)
    
    log.logger.info("正在join数据: id-query-features")
    index_df.sql_ctx.sparkSession._jsparkSession = spark._jsparkSession
    index_df._sc = spark._sc

    query_index_embedding_df = alsModel.userFactors
    query_index_embedding_df.sql_ctx.sparkSession._jsparkSession = spark._jsparkSession
    query_index_embedding_df._sc = spark._sc

    query_embedding_df = query_index_embedding_df.join(index_df, query_index_embedding_df.id == index_df.queryIndex).select("id", "keyword", "features")
    query_embedding_df.sql_ctx.sparkSession._jsparkSession = spark._jsparkSession
    query_embedding_df._sc = spark._sc

    # id_query_map = query_embedding_df.select("id", "keyword").rdd.collectAsMap() # 转map {id:query}
    # query_embedding_map = query_embedding_df.select("keyword", "features").rdd.collectAsMap() # 转map {query:embedding}
    # id_quEmbedding_map = query_embedding_df.select("id","features").rdd.collectAsMap() # 转map {id:embedding}

    log.logger.info(f"并保存向量文件: {query_filename}")
    query_embedding_df.distinct() \
                      .toPandas() \
                      .to_csv(query_filename, index=False, header=False) # faiss的训练数据

    log.logger.info("正在join数据: id-doc-features")
    doc_index_embedding_df = alsModel.itemFactors
    doc_index_embedding_df.sql_ctx.sparkSession._jsparkSession = spark._jsparkSession
    doc_index_embedding_df._sc = spark._sc
    
    doc_embedding_df = doc_index_embedding_df.join(index_df, doc_index_embedding_df.id == index_df.docIndex).select("id", "id_type", "features")
    doc_embedding_df.sql_ctx.sparkSession._jsparkSession = spark._jsparkSession
    doc_embedding_df._sc = spark._sc
    
    # id_doc_map = doc_embedding_df.select("id", "id_type").rdd.collectAsMap() # 转map {id:query}
    # doc_embedding_map = doc_embedding_df.select("id_type", "features").rdd.collectAsMap() # 转map {query:embedding}
    # id_docEmbedding_map = doc_embedding_df.select("id","features").rdd.collectAsMap() # 转map {id:embedding}

    log.logger.info(f"并保存向量文件: {doc_filename}")
    doc_embedding_df.distinct() \
                    .toPandas() \
                    .to_csv(doc_filename, index=False, header=False)  # faiss的训练数据
    
    # 清除缓存
    index_df.unpersist()
    spark.stop()  # 此行一定要加，否则有一天spark会内存耗尽无法启动

    # return query_embedding_map, doc_embedding_map, id_query_map, id_doc_map, id_quEmbedding_map, id_docEmbedding_map
    # 考虑保存为csv文件，使用时pandas, json读取


if __name__ == "__main__":
    print("开始构建ALS embedding ……")
    query_emb_path = "./data/query_embedding.csv"
    doc_emb_path = "./data/doc_embedding.csv"
    
    startTime = (datetime.datetime.now() + datetime.timedelta(days=-7)).strftime("%Y%m%d")
    endTime = (datetime.datetime.now() + datetime.timedelta(days=-1)).strftime("%Y%m%d")
    print(f"开始日期 {startTime}, 结束日期 {endTime}")

    # query_embedding_map, doc_embedding_map, id_query_map, id_doc_map, id_quEmbedding_map, id_docEmbedding_map = 
    get_als_embedding(startTime, endTime, query_emb_path, doc_emb_path)

    query_embedding_map = pd.read_csv(query_emb_path, usecols=[1, 2], index_col=0, squeeze=True, header=None).to_dict()
    print("北京 的向量: ", query_embedding_map["北京"])
    
    doc_embedding_map = pd.read_csv(doc_emb_path, usecols=[1, 2], index_col=0, squeeze=True, header=None).to_dict()

    id_query_map = pd.read_csv(query_emb_path, usecols=[0, 1], index_col=0, squeeze=True, header=None).to_dict()

    id_doc_map = pd.read_csv(doc_emb_path, usecols=[0, 1], index_col=0, squeeze=True, header=None).to_dict()

    id_quEmbedding_map = pd.read_csv(query_emb_path, usecols=[0, 2], index_col=0, squeeze=True, header=None).to_dict()

    id_docEmbedding_map = pd.read_csv(doc_emb_path, usecols=[0, 2], index_col=0, squeeze=True, header=None).to_dict()
    
    print("热搜前10的query: ")
    for i in range(10):
        print(id_query_map[i])

    print("\n点击前10的doc: ")
    for i in range(10):
        print(id_doc_map[i])
