import pickle
import numpy
from datasketch import MinHashLSHForest, MinHash
from tqdm import tqdm


forest = MinHashLSHForest(num_perm=64)
for id in tqdm(range(100)):
    hash_code = list(range(128))
    minhash = MinHash(hashvalues=numpy.array(hash_code))
    forest.add(id, minhash=minhash)

forest.index()

serialized_data = pickle.dumps(forest)
with open('xxx.pickle', 'wb') as f:
    f.write(serialized_data)
