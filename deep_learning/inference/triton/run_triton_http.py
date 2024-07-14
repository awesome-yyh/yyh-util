import time
import gevent.ssl
import numpy as np
import tritonclient.http as httpclient
from transformers import AutoModelForSeq2SeqLM, BertTokenizer, BertModel


class TritonClient:
    def __init__(self, url="localhost:8000",
                 ssl=False, key_file=None, cert_file=None, ca_certs=None, insecure=False, verbose=False):
        """
        所有模型通用的client
        :param url:
        :param ssl: Enable encrypted link to the server using HTTPS
        :param key_file: File holding client private key
        :param cert_file: File holding client certificate
        :param ca_certs: File holding ca certificate
        :param insecure: Use no peer verification in SSL communications. Use with caution
        :param verbose: Enable verbose output
        :return: triton_client
        """
        if ssl:
            ssl_options = {}
            if key_file is not None:
                ssl_options['keyfile'] = key_file
            if cert_file is not None:
                ssl_options['certfile'] = cert_file
            if ca_certs is not None:
                ssl_options['ca_certs'] = ca_certs
            ssl_context_factory = None
            if insecure:
                ssl_context_factory = gevent.ssl._create_unverified_context
            self.triton_client = httpclient.InferenceServerClient(
                url=url,
                verbose=verbose,
                ssl=True,
                ssl_options=ssl_options,
                insecure=insecure,
                ssl_context_factory=ssl_context_factory)
        else:
            self.triton_client = httpclient.InferenceServerClient(
                url=url, verbose=verbose)


class SentenceEmbInfer(TritonClient):
    def __init__(self, url, hf_model_path, model_name):
        super(SentenceEmbInfer, self).__init__(url)
        
        self.tokenizer = BertTokenizer.from_pretrained(hf_model_path)
        self.model_name = model_name
        
    def infer(self, sentences, model_version="1",
              input0='input_ids', input1='token_type_ids', input2='attention_mask',
              output0='sentence_embeddings',
              request_compression_algorithm=None,
              response_compression_algorithm=None):
        """

        :param triton_client:
        :param model_name:
        :param input0:
        :param input1:
        :param output0:
        :param output1:
        :param request_compression_algorithm: Optional HTTP compression algorithm to use for the request body on client side.
                Currently supports "deflate", "gzip" and None. By default, no compression is used.
        :param response_compression_algorithm:
        :return:
        """
        inputs = []
        outputs = []
        
        encoded_input = self.tokenizer(sentences, max_length=128, padding=True, truncation=True)
        print(encoded_input, type(encoded_input), len(encoded_input['input_ids'][0]))
        
        batch_size = len(encoded_input['input_ids'])
        max_length = len(encoded_input['input_ids'][0])
        # 如果batch_size超过配置文件的max_batch_size，infer则会报错
        # INPUT0、INPUT1为配置文件中的输入节点名称
        inputs.append(httpclient.InferInput(input0, [batch_size, max_length], "INT32"))
        inputs.append(httpclient.InferInput(input1, [batch_size, max_length], "INT32"))
        inputs.append(httpclient.InferInput(input2, [batch_size, max_length], "INT32"))
        
        inputs[0].set_data_from_numpy(np.array(encoded_input["input_ids"], dtype=np.int32), binary_data=False)
        inputs[1].set_data_from_numpy(np.array(encoded_input["token_type_ids"], dtype=np.int32), binary_data=False)
        inputs[2].set_data_from_numpy(np.array(encoded_input["attention_mask"], dtype=np.int32), binary_data=False)

        # OUTPUT0、OUTPUT1为配置文件中的输出节点名称
        outputs.append(httpclient.InferRequestedOutput(output0, binary_data=False))

        # query_params = {'test_1': 1, 'test_2': 2}
        results = self.triton_client.infer(
            model_name=self.model_name,
            model_version=model_version,
            inputs=inputs,
            outputs=outputs,
            # query_params=query_params,
            headers=None,
            request_compression_algorithm=request_compression_algorithm,
            response_compression_algorithm=response_compression_algorithm)
        
        # 转化为numpy格式
        return results.as_numpy(output0)

    def info(self):
        model_repository_index = self.triton_client.get_model_repository_index()
        print("model_repository_index: ", model_repository_index)
        server_meta = self.triton_client.get_server_metadata()
        print("server_meta: ", server_meta)
        model_meta = self.triton_client.get_model_metadata(self.model_name)
        print("model_meta: ", model_meta)
        model_config = self.triton_client.get_model_config(self.model_name)
        print("model_config: ", model_config)
        statistics = self.triton_client.get_inference_statistics()
        print("statistics: ", statistics)
        shm_status = self.triton_client.get_cuda_shared_memory_status()
        print("shm_status: ", shm_status)
        sshm_status = self.triton_client.get_system_shared_memory_status()
        print("sshm_status: ", sshm_status)
        
        server_live = self.triton_client.is_server_live()
        print("server_live: ", server_live)
        server_ready = self.triton_client.is_server_ready()
        print("server_ready: ", server_ready)
        model_ready = self.triton_client.is_model_ready(self.model_name)
        print("model_ready: ", model_ready)

    def load_model(self,):
        self.triton_client.load_model(self.model_name)
    
    def unload_model(self,):
        self.triton_client.unload_model(self.model_name)


if __name__ == "__main__":
    hf_model = "/data/app/base_model/shibing624-text2vec-base-chinese"
    client = SentenceEmbInfer(url="10.96.1.34:8000", hf_model_path=hf_model, model_name="sentence_emb")
    client.info()
    client.unload_model()
    client.load_model()
    
    start_time = time.time()
    sentences = ['10月组织四川平民学社，并出版刊物《》。', '爝光', '曙光']
    print(client.infer(sentences=sentences))
    print(time.time() - start_time, ' s')
