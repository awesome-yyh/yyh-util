'''
Author: yyh owyangyahe@126.com
Date: 2024-09-27 11:02:24
LastEditors: yyh owyangyahe@126.com
LastEditTime: 2024-09-27 17:15:23
FilePath: /mypython/yyh-util/py/thread.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import asyncio
import time
import aiohttp
import requests
import json


# url = "http://127.0.0.1:80/items"
url = "http://127.0.0.1:80/items_async"


headers = {
  'Content-Type': 'application/json'
}


# def post_api(item=0):
#     payload = json.dumps({
#         "id": item
#     })
#     response = requests.request("POST", url, headers=headers, data=payload)
#     return response.text


# async def post_api(item=0):
#     payload = json.dumps({
#         "id": item
#     })
#     response = requests.request("POST", url, headers=headers, data=payload)  # requests 只能发送同步请求，发起请求后在收到结果之前不能发起下一次请求
#     return response.text


async def post_api(item=0):
    payload = json.dumps({
        "id": item
    })
    async with aiohttp.ClientSession() as session:  # aiohttp 只能发送异步请求, 在等待结果的时间里可以继续发送更多请求
        async with session.post(url, headers=headers, data=payload) as response:  
            return await response.text()  

async def main():  
    items = list(range(10))
    print(items)
    tasks = [post_api(item) for item in items]
    print(tasks)
    t = asyncio.gather(*tasks)  # 同时执行多个异步任务
    print(t, type(t))
    return await t  # 要异步结果

async def my_main():
    return await asyncio.gather(post_api(), main())

if __name__ == "__main__":
    start_time = time.time()
    print(asyncio.run(post_api()))
    print(asyncio.run(main()))
    print(f"耗时: {time.time() - start_time}")
    
    start_time = time.time()
    print(asyncio.run(my_main()))
    print(f"耗时: {time.time() - start_time}")