'''
Author: yyh owyangyahe@126.com
Date: 2024-09-27 11:02:24
LastEditors: yyh owyangyahe@126.com
LastEditTime: 2024-09-27 14:29:43
FilePath: /mypython/yyh-util/py/thread.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time
import requests
import json


url = "http://127.0.0.1:80/items"

headers = {
  'Content-Type': 'application/json'
}


def post_api(item=0):
    payload = json.dumps({
        "id": item
    })
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.text



if __name__ == "__main__":
    print(post_api())
    start_time = time.time()
    items = list(range(10))
    with ThreadPoolExecutor(max_workers=1024, thread_name_prefix='src_batch') as pool:
        results = pool.map(post_api, items)
    print(list(results))
    print(f"耗时: {time.time() - start_time}")