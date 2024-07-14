import datetime
import json
import re
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm


class ES():
    """ES类
    typical usage example:

    :traffic_ES = ES(host_key=["xx.xx.xx.xx:xxxx"], index_name="xxx")
    :doc_dict, return_field_dict = traffic_ES.get_es_data(DSL, ["doc_id"])

    Args:

    :host_key(list): ES集群的各个域名, 元素为字符串: "xx.xx.xx.xx:xxxx" (ip或域名:端口)
    :index_name(str): 索引名
    """
    def __init__(self, host_key, index_name):
        self.es = Elasticsearch(host_key)
        # 在进行任何操作前先进行嗅探，节点没有响应时刷新重新连接，每60秒刷新一次
        # self.es = Elasticsearch(host_key, timeout=60, sniff_on_start=True, sniff_on_connection_fail=True, sniffer_timeout=60)
        # self.es = Elasticsearch([
        #     {'host': host_key, 'port': 443, 'url_prefix': 'es', 'use_ssl': True},
        # ])
        
        if self.es.ping():
            print("es初始化完成")
        else:
            print("es初始化失败")
        
        self.index_name = index_name
    
    def get_datas(self, body, return_fields=[]):
        """查询ES
        用scroll的方式获取一个index-type的大量数据，未考虑内存控制

        Args:

        :body(dict):  查询的DSL
        :return_fields(list): 需要返回的字段名列表
        """
        assert isinstance(body, dict), u"body 必须是一个 dict类型"
        assert "query" in body, f"body 中必须含有 query 这个查询体,body: {json.dumps(body)}"
        body["_source"] = {"include": return_fields}
        res = self.es.search(index=self.index_name, body=body, scroll="1m", request_timeout=30)  # 30 seconds
        total_num = 0
        if isinstance(res["hits"]["total"], int):  # ES5
            total_num = res["hits"]["total"]
        elif isinstance(res["hits"]["total"], dict):  # ES7
            total_num = res["hits"]["total"]['value']
        else:
            print("请检查ES版本不同导致此处的不兼容")
        
        print(f"共查询到 {total_num} 条数据")
        now_num = 0

        doc_dict = dict()
        return_field_dict = dict()
        for field in return_fields:
            return_field_dict[field] = []
        return_field_dict["id"] = []
        if total_num == 0:
            return doc_dict, return_field_dict

        while True:
            print(f"正在查询索引: {self.index_name}, 进度: {100.0 * now_num / total_num}----")
            assert not res["timed_out"], "已经超时，本次传输基本失败"
            __scroll_id = res["_scroll_id"]
            max_score = res["hits"]["max_score"]
            for item_doc in res["hits"]["hits"]:
                _id = item_doc["_id"]
                _source = item_doc["_source"]
                _source["_score"] = item_doc["_score"]  # 添加元素-score
                _source["max_score"] = max_score  # 添加元素-max_score
                now_num += 1
                doc_dict[_id] = _source
                # 返回字段
                for field in return_fields:
                    if field in _source:
                        return_field_dict[field].append(_source[field])
                return_field_dict["id"].append(_id)

            # bulk_data(doc_list, "online_qp_cache2", "qp_cache2")
            # doc_list = [] # 边读边批量处理
            res = self.es.scroll(scroll_id=__scroll_id, scroll="1m")
            if len(res["hits"]["hits"]) == 0:
                break
        assert total_num == now_num, f"total_num({total_num})和now_num({now_num})必须一致"
        return doc_dict, return_field_dict

    def divide_chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i: i + n]
    
    def put_datas(self, doc_list, max_chunk=1000):
        """更新ES
        既能更新，也能新建不存在的doc和字段

        Args:
            doc_list(list), 元素是字典（key为要更新的字段名，value为要更新的值）
                    该字典必须含'id'这个key
        """
        if len(doc_list) == 0:
            print('要更新的实时数据为空')
            return

        tag_doc_ = []
        print("正在更新es……", self.index_name)
        for chunk in tqdm(self.divide_chunks(doc_list, max_chunk)):
            for item in chunk:
                assert isinstance(item, dict), u"doc 必须是一个 dict类型"
                assert "id" in item, u"doc 必须含'id'这个key"
                
                chunk_all = dict()
                chunk_all["doc"] = item
                chunk_all["doc_as_upsert"] = "true"
                chunk_all["_index"] = self.index_name
                chunk_all["_type"] = self.index_name
                chunk_all["_id"] = item["id"]
                chunk_all["_op_type"] = "update"

                tag_doc_.append(chunk_all)

            sucess_num, error_list = helpers.bulk(self.es, tag_doc_)  # (client  doc集合list)
            if sucess_num != len(doc_list):
                print("验证文件行必须和ES成功bulk数目必须一致，现在不是")
                print(f"build error {error_list}")

            print("本次成功更新目标文档数: " + str(sucess_num))


if __name__ == "__main__":
    host_key = "xxxx:xx"
    policy_es = ES(host_key=[host_key], index_name="xxxxxx")
    dsl = {
        "query": {}
    }
    doc_dict, return_field_dict = policy_es.get_datas(dsl, ["DreTitle"])
    print(f"共查询到 {len(doc_dict)} 条记录\n", return_field_dict, doc_dict)
    # print(policy_es.get_datas(dsl, ["DelFlag"]))
    
    file = "es/train.json"
    with open(file, "r") as f:
        data = json.load(f)
    
    doc_list = []
    for index, item in enumerate(data):
        doc_i = dict()
        doc_i["id"] = index
        doc_i.update(item)
        
        doc_list.append(doc_i)
        
    print("总共要更新: ", len(doc_list))
    policy_es.put_datas(doc_list)
    