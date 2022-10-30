from pyspark.sql import SparkSession

# SparkSession是整个spark应用的起点
# appName 是在yarn管理界面查看的应用名称
spark = SparkSession.builder \
      .master("local") \
      .appName("yyhALS") \
      .config("spark.submit.deployMode","client") \
      .config("spark.port.maxRetries", "100") \
      .config("spark.sql.broadcastTimeout", "1200") \
      .config("spark.yarn.queue", "root.search") \
      .enableHiveSupport() \
      .getOrCreate()

df = spark.read.options(delimiter=',') \
  .csv('/Users/yaheyang/sample.csv', header=False)

df.write.mode("overwrite").format("tfrecord").option("recordType", "Example").save("/Users/yaheyang/ss") # spark 写tfrecord

# spark-submit --class xxx.xx.xx.mainObject  --master local[2]   /opt/xxx.jar
