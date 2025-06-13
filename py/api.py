'''
Author: yyh owyangyahe@126.com
Date: 2024-09-27 10:50:06
LastEditors: yyh owyangyahe@126.com
LastEditTime: 2024-11-09 09:44:47
FilePath: /mypython/yyh-util/py/api.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import asyncio
import time
from typing import Union
from fastapi import FastAPI, requests
import fire
import uvicorn
from pydantic import BaseModel


class items(BaseModel):
    id: int

def main(port=80):
    app = FastAPI()
    print("api okkk")
    
    @app.get('/')
    async def document():
        return {"hello": "world"}

    @app.post("/items_async")
    async def read_item_async(param: items):
        print(f"item_id: {param.id}")
        await asyncio.sleep(2)  # 使用异步的 sleep，多个请求进来，可以同时执行这个
        # time.sleep(2)  # 这种，多个请求同时进来，也是一个一个完成，造成阻塞
        return {"async_item_id": param.id}
    
    @app.post("/items")
    def read_item(requests: requests):
        """默认异步

        Args:
            param (items): _description_

        Returns:
            _type_: _description_
        """
        print(f"item_id: {requests.json()}")
        time.sleep(2)
        return {"item_id": requests.json()}
    
    uvicorn.run(app=app, host="0.0.0.0", port=port, log_level="error", workers=1)
    
if __name__ == "__main__":
    fire.Fire(main)