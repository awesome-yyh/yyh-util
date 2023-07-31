import time
import numpy as np
import tritonclient.grpc as grpcclient


if __name__ == "__main__":

    triton_client = grpcclient.InferenceServerClient(
        url='10.96.1.34:8001',
        verbose=False,
        ssl=False,
        root_certificates=None,
        private_key=None,
        certificate_chain=None)
   
    inputs = []
    inputs.append(grpcclient.InferInput('input_ids', [3, 21], "INT32"))
    inputs.append(grpcclient.InferInput('token_type_ids', [3, 21], "INT32"))
    inputs.append(grpcclient.InferInput('attention_mask', [3, 21], "INT32"))
    
    inputs[0].set_data_from_numpy(np.array([[101, 8108, 3299, 5299, 5302, 1724, 2335, 2398, 3696, 2110, 4852, 8024, 2400, 1139, 4276, 1149, 4289, 517, 518, 511, 102], [101, 100, 1045, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 3282, 1045, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).astype(np.int32))
    
    inputs[1].set_data_from_numpy(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).astype(np.int32))
    
    inputs[2].set_data_from_numpy(np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).astype(np.int32))
    
    outputs = []
    outputs.append(grpcclient.InferRequestedOutput('sentence_embeddings'))

    start_time = time.time()
    results = triton_client.infer(model_name='sentence_emb', inputs=inputs, outputs=outputs, compression_algorithm=None)

    result = results.as_numpy("sentence_embeddings")
    print(result)
    print(time.time() - start_time, ' s')

