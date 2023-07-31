import time
import requests
import numpy as np


if __name__ == "__main__":
    request_data = {
        "inputs": [{
            "name": "input_ids",
            "shape": [3, 21],
            "datatype": "INT32",
            "data": [[101, 8108, 3299, 5299, 5302, 1724, 2335, 2398, 3696, 2110, 4852, 8024, 2400, 1139, 4276, 1149, 4289, 517, 518, 511, 102], [101, 100, 1045, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 3282, 1045, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        },
            {
            "name": "token_type_ids",
            "shape": [3, 21],
            "datatype": "INT32",
            "data": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        },
            {
            "name": "attention_mask",
            "shape": [3, 21],
            "datatype": "INT32",
            "data": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        }],
        "outputs": [{"name": "sentence_embeddings"}]
    }
    
    start_time = time.time()
    res = requests.post(url="http://10.96.1.34:8000/v2/models/sentence_emb/versions/1/infer", json=request_data).json()
    print(res)
    print(time.time() - start_time, ' s')
