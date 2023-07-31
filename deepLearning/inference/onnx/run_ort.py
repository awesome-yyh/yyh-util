import time
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
import onnxruntime as ort
import pred2label


class OrtInference():
    def __init__(self, onnx_path, base_model_path, max_len) -> None:
        self.ort_session = ort.InferenceSession(onnx_path, providers=ort.get_available_providers())
        
        self.tokenizer = BertTokenizer.from_pretrained(base_model_path)
        self.max_len = max_len
    
    def dataset(self, sentences):
        token_dict = self.tokenizer(
            sentences,  # [cls]句子1[sep]句子2[sep]
            padding='max_length',  # 不足时填充到指定最大长度
            truncation=True,  # 过长时截断
            max_length=self.max_len,  # 2个句子加起来的长度
            return_tensors='pt')  # 返回字典, input_ids, token_type_ids, attention_mask
    
        ort_inputs = {input_name: value.numpy() for input_name, value in token_dict.items()}
        
        return ort_inputs

    def inference(self, sentences, is_need_prob=False):
        ort_inputs = self.dataset(sentences)
        preds = self.ort_session.run(None, ort_inputs)[0][0]
        
        preds = pred2label.pred2label_one(preds, is_need_prob)
        # print("====preds====:", preds)
        return preds

    def inference_only(self, ort_inputs):
        preds = self.ort_session.run(None, ort_inputs)[0][0]
        return preds
    
    def io_info(self):
        # 查看模型的输入输出信息
        [print(inputs) for inputs in self.ort_session.get_inputs()]
        [print(inputs) for inputs in self.ort_session.get_outputs()]


if __name__ == "__main__":
    onnx_path = "checkpoints/onnx/model_best.onnx"
    # onnx_path = "checkpoints/onnx/model_best_fp16.onnx"
    base_model_path = "/data/app/base_model/chinese-roberta-wwm-ext"
    max_len = 180
    ort_infer = OrtInference(onnx_path, base_model_path, max_len)
    ort_infer.io_info()
    
    sample_text = ["吃葡萄不吐葡萄皮"]
    
    steps = 1000
    ort_inputs = ort_infer.dataset(sample_text)
    start = time.time()
    for i in tqdm(range(steps)):
        # label = ort_infer.inference(sample_text)[0]
        label = ort_infer.inference_only(ort_inputs)[0]
    print('onnxruntime(不含构造数据) 平均耗时: ', (time.time() - start) * 1000 / steps, ' ms')
