# -*- coding: utf-8 -*-
import redis

class RedisSingle:
  """Redis单机类
  typical usage example:

  :host = 'x.xx.xx.xx'
  :redis36 = RedisSingle(host)
  
  :redis36.set('yyh001', "yyhyyh")
  :print("redis 写完成")
  
  :print("redis 读完成, 结果是：" + redis36.get("yyh001"))

  Args:

  host(str): 要连接的Redis ip, 例: xx.xx.xx.xx
  port(int): 要连接的Redis 端口
  """
  def __init__(self, host, port=6379):
    try:
        pool = redis.ConnectionPool(
            host=host, port=port, decode_responses=True)
        self.__redis = redis.Redis(connection_pool=pool)
        print("redis single connected success, ip: " + str(host))
        
    except:
        print("could not connect to redis single.")

  def get(self, cacheKey):
    """读Redis

    Args:
        cacheKey (str): redis key

    Returns:
        str: 读取的结果
    """
    return self.__redis.get(cacheKey)

  def set(self, cachekey, cacheValue, ex=60*60):
    """写Redis

    Args:
        cachekey (str): redis key
        cacheValue (str): redis value
        ex (str, optional): 过期时间. Defaults to 60*60.
    """
    self.__redis.set(cachekey, cacheValue)
