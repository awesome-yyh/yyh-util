import json
import datetime
from elasticsearch import Elasticsearch, helpers

class Es():
  """ES类
  typical usage example:

  :trafficEs = Es(host_key=["xx.xx.xx.xx:xxxx"], index_name="xxx",
                                          type_name='xxx')
  :docId_list, doc_dict, returnField_dict = trafficEs.getEsData(cdeDSL, "doc_id")

  Args:

  :host_key(list): ES集群的各个域名, 元素为字符串: "xx.xx.xx.xx:xxxx" (ip或域名:端口)
  :index_name(str): 索引名
  :type_name(str): 类型名
  """
  def __init__(self, host_key, index_name, type_name):
    self.index_name = index_name
    self.type_name = type_name
    # 在进行任何操作前先进行嗅探，节点没有响应时刷新重新连接，每60秒刷新一次
    self.es = Elasticsearch(host_key, timeout=60, sniff_on_start=True,
                        sniff_on_connection_fail=True, sniffer_timeout=60)
  
  def getEsData(self, body, *returnFields):
    """获取ES
    用scroll的方式获取一个index-type的大量数据，未考虑内存控制

    Args:

    :body(dict):  查询的DSL
    :returnFields(str): 需要返回的字段
    """
    assert isinstance(body, dict), u"body 必须是一个 dict类型"
    assert "query" in body, u"body 中必须含有 query 这个查询体,body: {a}".format(
        a=json.dumps(body))
    res = self.es.search(index=self.index_name, doc_type=self.type_name,
                     body=body, scroll="1m")
    total_num = res["hits"]["total"]
    now_num = 0
    
    docId_list = []
    doc_dict = dict()
    returnField_dict = dict()
    for returnField in returnFields:
      returnField_dict[returnField] = []
    if total_num == 0:
        return docId_list, doc_dict, returnField_dict

    while True:
        print("now process {}----".format(100.0 * now_num / total_num))
        assert res["timed_out"] == False, "已经超时，本次传输基本失败"
        __scroll_id = res["_scroll_id"]
        for item_doc in res["hits"]["hits"]:
            _source = item_doc["_source"]
            _id = item_doc["_id"]
            now_num += 1
            # 返回字段
            docId_list.append(_id)
            doc_dict[_id] = _source
            for returnField in returnFields:
              returnField_dict[returnField].append(_source[returnField])

        # bulk_data(doc_list, "online_qp_cache2", "qp_cache2")
        # doc_list = [] # 边读边批量处理
        res = self.es.scroll(scroll_id=__scroll_id, scroll="1m")
        if len(res["hits"]["hits"]) == 0:
            break
    assert total_num == now_num, u"total_num({a})和now_num({b})必须一致".format(
        a=total_num, b=now_num
    )
    return docId_list, doc_dict, returnField_dict

