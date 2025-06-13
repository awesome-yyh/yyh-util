from rediscluster import StrictRedisCluster


class RedisCluster:
    """Redis集群读写类
    typical usage example:

    :redisContent = RedisCluster(conn_list)
    :print("正在进行写入")
    :redisContent.set("yyhh001", "yyhh002")
    :print("正在进行读取")
    :print("读取完成：" + redisContent.get("yyhh001"))

    Args:

    conn_list(list): 要连接的Redis集群ip及端口, 例：[{'host': 'xx.xx.xx.xx', 'port': 7000}]
    """
    def __init__(self, conn_list):
        try:
            # 非密码连接redis集群
            self.__redis = StrictRedisCluster(
                startup_nodes=conn_list, decode_responses=True)
            # 使用密码连接redis集群
            # self.redis = StrictRedisCluster(startup_nodes=conn_list, password='123456')
            print("redis 集群连接成功")
        except Exception as e:
            print(e)
            print("错误,连接redis 集群失败")

    def get(self, cacheKey):
        """读Redis

        Args:
            cacheKey (str): redis key

        Returns:
            str: 读取的结果
        """
        return self.__redis.get(cacheKey)

    def set(self, cachekey, cacheValue, ex=60 * 60):
        """写Redis

        Args:
            cachekey (str): redis key
            cacheValue (str): redis value
            ex (str, optional): 过期时间. Defaults to 60*60.
        """
        self.__redis.set(cachekey, cacheValue, ex)
