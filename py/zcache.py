from pylru import lrucache
from typing import List
import hashlib


class ZCache():
    """
    适用于模型等并行计算场景
    输入一个list，有的在cache中，有的不在，
    找出不在cache中，以bache形式输入[模型]并行计算
    按原顺序合并所有结果，并把不在cache中的存到cache中
    """
    def __init__(self, maxsize=2**16):
        self.cache = lrucache(maxsize)
    
    def _make_key(self, sentence):
        return sentence if len(sentence) < 16 else hashlib.md5(sentence.encode('utf-8')).hexdigest()
    
    def calc_with_cache(self, sentences: List[str], calc_func):
        cached_results: List[str] = []  # 已缓存的结果
        uncached_sentences: List[str] = []  # uncached sentences
        idx_iscache: List[bool] = []  # 各句子是否有缓存
        idx_mapping: List[int] = []  # 各句子在cached_results和uncached_sentences中的下标
        
        # setp1: filter
        for sentence in sentences:
            key = self._make_key(sentence)
            if key in self.cache:
                cached_results.append(self.cache[key])
                idx_iscache.append(True)
                idx_mapping.append(len(cached_results) - 1)
            else:
                uncached_sentences.append(sentence)
                idx_iscache.append(False)
                idx_mapping.append(len(uncached_sentences) - 1)
        
        if len(uncached_sentences) == 0:
            return cached_results

        # setp2: calc
        # print("uncached_sentences: ", uncached_sentences)
        calc_results = calc_func(uncached_sentences)
        
        # setp3: merge and save
        calc_result_all = self._merge_and_save(cached_results, uncached_sentences, idx_mapping, idx_iscache, calc_results)
        
        return calc_result_all

    def _merge_and_save(self, cached_results, uncached_sentences, idx_mapping, idx_iscache, calc_results):
        # 合并缓存与模型结构
        calc_result_all: List[str] = []
        try:
            for idx, cache_flag in zip(idx_mapping, idx_iscache):
                if cache_flag:
                    calc_result_all.append(cached_results[idx])
                else:
                    calc_result_all.append(calc_results[idx])
                    
                    self.cache[self._make_key(uncached_sentences[idx])] = calc_results[idx]  # 缓存新结果
        except Exception as e:
            print("缓存或合并结果出错：{}".format(str(e)))
            return

        return calc_result_all


if __name__ == "__main__":
    zcache = ZCache()
    
    def myfunc(ss: List[str]):
        return [s + "s" for s in ss]
    
    sentences = ["1", "2", "3"]
    calc_result_all = zcache.calc_with_cache(sentences, calc_func=myfunc)
    print(calc_result_all)
    
    sentences = ["1", "2", "3", "6"]
    calc_result_all = zcache.calc_with_cache(sentences, calc_func=myfunc)
    print(calc_result_all)
    