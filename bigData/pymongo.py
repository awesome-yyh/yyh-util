import pymongo
from tqdm import tqdm


myclient = pymongo.MongoClient(
    "mongodb://localhost:27017/"
)
print(myclient.list_database_names())

mydb = myclient["database"]
print(mydb.list_collection_names())

collection = mydb["collection"]

print(collection.find_one())

for result in tqdm(collection.find({}), total=2000 * 10000):  # find({}, {}) # 查询条件, 该字段是否返回
    id = str(result['_id'])
    content = result['content']
    print("==>", id, content)
    break
