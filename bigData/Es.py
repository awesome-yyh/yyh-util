import json
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

class Es():
  """ES类
  typical usage example:

  :trafficEs = Es(host_key=["xx.xx.xx.xx:xxxx"], index_name="xxx")
  :docId_list, doc_dict, returnField_dict = trafficEs.getEsData(cdeDSL, "doc_id")

  Args:

  :host_key(list): ES集群的各个域名, 元素为字符串: "xx.xx.xx.xx:xxxx" (ip或域名:端口)
  :index_name(str): 索引名
  """
  def __init__(self, host_key, index_name):
    self.index_name = index_name
    # 在进行任何操作前先进行嗅探，节点没有响应时刷新重新连接，每60秒刷新一次
    self.es = Elasticsearch(host_key, timeout=60, sniff_on_start=True,
                        sniff_on_connection_fail=True, sniffer_timeout=60)
  

  def getEsData(self, body):
    """获取ES
    用scroll的方式获取一个index-type的大量数据，未考虑内存控制

    Args:

    :body(dict):  查询的DSL
    :returnFields(str): 需要返回的字段
    """
    assert isinstance(body, dict), u"body 必须是一个 dict类型"
    assert "query" in body, u"body 中必须含有 query 这个查询体,body: {a}".format(
        a=json.dumps(body))
    res = self.es.search(index=self.index_name,
                     body=body, scroll="1m", request_timeout=30) # 30 seconds
    total_num = 0
    if isinstance(res["hits"]["total"], int): # ES5
      total_num = res["hits"]["total"]
    elif isinstance(res["hits"]["total"], dict):  # ES7
      total_num = res["hits"]["total"]['value']
    else:
      print("请检查ES版本不同导致此处的不兼容")
    
    print(f"共查询到 {total_num} 条数据")
    now_num = 0

    doc_dict_list = []
    if total_num == 0:
        return doc_dict_list

    while True:
        print("正在查询索引: {}, 进度: {}----".format(self.index_name, 100.0 * now_num / total_num))
        assert res["timed_out"] == False, "已经超时，本次传输基本失败"
        __scroll_id = res["_scroll_id"]
        # max_score = res["hits"]["max_score"]
        # print(str(res["hits"]["hits"]).replace('\'', '\"'))
        doc_dict_list.extend(res["hits"]["hits"])
        # for item_doc in res["hits"]["hits"]:
        #     _id = item_doc["_id"]
        #     _source = item_doc["_source"]
        #     _source["_score"] = item_doc["_score"] # 添加元素-score
        #     _source["max_score"] = max_score # 添加元素-max_score
        #     now_num += 1
        #     # 返回字段
        #     doc_dict[_id] = _source
        res = self.es.scroll(scroll_id=__scroll_id, scroll="1m")
        if len(res["hits"]["hits"]) == 0:
            break
    # assert total_num == now_num, u"total_num({a})和now_num({b})必须一致".format(
    #     a=total_num, b=now_num
    # )
    return doc_dict_list

  def getEsData2(self, body, *returnFields):
    """获取ES
    用scroll的方式获取一个index-type的大量数据，未考虑内存控制

    Args:

    :body(dict):  查询的DSL
    :returnFields(str): 需要返回的字段
    """
    assert isinstance(body, dict), u"body 必须是一个 dict类型"
    assert "query" in body, u"body 中必须含有 query 这个查询体,body: {a}".format(
        a=json.dumps(body))
    res = self.es.search(index=self.index_name,
                     body=body, scroll="1m", request_timeout=30) # 30 seconds
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
        print("正在查询索引: {}, 进度: {}----".format(self.index_name, 100.0 * now_num / total_num))
        assert res["timed_out"] == False, "已经超时，本次传输基本失败"
        __scroll_id = res["_scroll_id"]
        max_score = res["hits"]["max_score"]
        for item_doc in res["hits"]["hits"]:
            _id = item_doc["_id"]
            _source = item_doc["_source"]
            _source["_score"] = item_doc["_score"] # 添加元素-score
            _source["max_score"] = max_score # 添加元素-max_score
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


  def _setEsData1000(self, docList):
    if len(docList) == 0:
        print('要更新的实时数据为空')
        return
    assert len(docList) <= 2000, "更新太多了，es承受不住"

    tag_doc_ = []
    print("正在更新es……", self.index_name)
    for item in docList:
      assert isinstance(item, dict), u"doc 必须是一个 dict类型"
      assert "id" in item, u"doc 必须含'id'这个key"
      _id = item["id"]

      chunk_all = dict()
      chunk_all["doc"] = item
      chunk_all["doc_as_upsert"] = "true"
      chunk_all["_index"] = self.index_name
      chunk_all["_id"] = _id
      chunk_all["_op_type"] = "update"

      tag_doc_.append(chunk_all)

    sucess_num, error_list = helpers.bulk(self.es, tag_doc_)  # (client  doc集合list)
    if sucess_num != len(docList):
      print("验证文件行必须和ES成功bulk数目必须一致，现在不是")
    if error_list:
      process_msg = "build error {ids}".format(ids=str(error_list))
      print(process_msg)

    print("本次更新目标文档数: " + str(sucess_num))

  def getDocList(self, **field): 
    """_summary_
    将不定数量个字段list组织成dict, dict的元素包括es的id等各个字段
    最后再把dict放到list中

    Args:
        field (dict): 关键字参数，值是list

    Returns:
        list: list中是一个一个的字典，字典有id等各个元素
    """
    docList = []
    value0 = next(iter(field.values())) # 取出第一个元素的值
    for item in range(len(value0)): # 元素的值是list，遍历其长度
      docDict = dict()
      for fieldName, fieldList in field.items():
        docDict[fieldName] = fieldList[item]
      docList.append(docDict)
    return docList

  def setEsData(self, docList):
    """更新ES
    既能更新，也能新建不存在的doc和字段

    Args:
        docList(list), 元素是字典（key为要更新的字段名，value为要更新的值）
                该字典必须含'id'这个key
    """
    num = 0
    docList1000 = []
    for i_doc in tqdm(docList):
      num += 1
      docList1000.append(i_doc)
      if num >= 1000:
        self._setEsData1000(docList1000)
        num = 0
        docList1000 = []
    self._setEsData1000(docList1000) # 剩余的