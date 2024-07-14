import os
import time
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
import tensorrt as trt
import nvcommon as nvcommon
import pred2label as pred2label
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# 设置gpu卡号时采用os.environ['CUDA_VISIBLE_DEVICES]='2'必须放置在pycuda.autoinit之前。
try:
    import pycuda.autoprimaryctx
except ModuleNotFoundError:
    import pycuda.autoinit


class TrtInference():
    def __init__(self, trt_path, base_model_path, max_len):
        # 读取.engine文件，并反序列化
        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(trt_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.inputs, self.outputs, self.bindings, self.stream = nvcommon.allocate_buffers(self.engine)
        self.context = self.engine.create_execution_context()
        
        self.tokenizer = BertTokenizer.from_pretrained(base_model_path)
        self.max_len = max_len

    def dataset(self, sentences):
        token_dict = self.tokenizer(
            sentences,  # [cls]句子1[sep]句子2[sep]
            padding='max_length',  # 不足时填充到指定最大长度
            truncation=True,  # 过长时截断
            max_length=self.max_len,  # 2个句子加起来的长度
            return_tensors='pt')  # 返回字典, input_ids, token_type_ids, attention_mask
    
        trt_inputs = {input_name: value.squeeze().numpy().astype(np.int32) for input_name, value in token_dict.items()}
        return trt_inputs

    def inference(self, sentences, is_need_prob=False):
        token_dict = self.dataset(sentences)
        self.inputs[0].host = token_dict["input_ids"]
        self.inputs[1].host = token_dict["token_type_ids"]
        self.inputs[2].host = token_dict["attention_mask"]

        preds = nvcommon.do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)[0]
        
        preds = pred2label.pred2label_one(preds, is_need_prob)
        # print("====preds====:", preds)
        return preds
    
    def inference_only(self, token_dict):
        self.inputs[0].host = token_dict["input_ids"]
        self.inputs[1].host = token_dict["token_type_ids"]
        self.inputs[2].host = token_dict["attention_mask"]

        preds = nvcommon.do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)[0]
        
        return preds
    
    def io_info(self):
        # 查看模型的输入输出信息，看是否与ONNX模型一致
        for idx in range(self.engine.num_bindings):
            is_input = self.engine.binding_is_input(idx)
            name = self.engine.get_binding_name(idx)
            op_type = self.engine.get_binding_dtype(idx)
            shape = self.engine.get_binding_shape(idx)

            print('input id:', idx, '   is input: ', is_input, '  binding name:', name, '  shape:', shape, 'type: ', op_type)


if __name__ == "__main__":
    trt_path = "checkpoints/trt/bestfp16.trt"
    # trt_path = "checkpoints/trt/best.trt"
    base_model_path = "/data/app/base_model/chinese-roberta-wwm-ext"
    max_len = 180
    trt_infer = TrtInference(trt_path, base_model_path, max_len)
    trt_infer.io_info()
    
    sample_text = ["吃葡萄不吐葡萄皮"]
    
    steps = 1000
    token_dict = trt_infer.dataset(sample_text)
    start = time.time()
    for i in tqdm(range(steps)):
        # label = trt_infer.inference(sample_text)[0]
        label = trt_infer.inference_only(token_dict)[0]
    print('tensorrt(不含构造数据) 平均耗时: ', (time.time() - start) * 1000 / steps, ' ms')
