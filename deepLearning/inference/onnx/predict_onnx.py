import time
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
import yaml
import onnx
import onnxruntime as ort


class OrtInference():
    def __init__(self, onnx_path, config_file='./configs/train.yaml') -> None:
        with open(config_file, 'r') as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.onnx_path = onnx_path
        # 创建session
        self.ort_session = ort.InferenceSession(onnx_path, providers=ort.get_available_providers())
    
    def dataset(self, question, context):
        tokenizer = BertTokenizer.from_pretrained(self.cfg["base_model_path"])
        
        inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, max_length=self.cfg["max_len"], truncation="only_second", padding='max_length', return_tensors='pt')
        
        ort_inputs = {input_name: value.numpy() for input_name, value in inputs.items()}
        
        return ort_inputs

    # def inference(self, sentences, is_need_prob=False):
    #     ort_inputs = self.dataset(sentences)
    #     preds = self.ort_session.run(None, ort_inputs)[0][0]
        
    #     preds = pred2label.pred2label_one(preds, is_need_prob)
    #     # print("====preds====:", preds)
    #     return preds

    def inference_only(self, ort_inputs):
        preds = self.ort_session.run(None, ort_inputs)
        print(preds, preds[0].shape, preds[1].shape)
        start_end_pos = self.post_pro(preds)
        print(start_end_pos)
        
        return start_end_pos
    
    def post_pro(self, preds):
        index_list = []
        for ipreds in preds:
            # print(np.argmax(ipreds[0]))
            # index_list.append(np.argmax(ipreds[0]))
            print(np.where(ipreds[0] >= 0))
            if np.where(ipreds[0] >= 0)[0].size != 0:
                index_list.append(np.where(ipreds[0] >= 0)[0].tolist()[0])
        return index_list
    
    def check_onnx(self, ):
        onnx_model = onnx.load(self.onnx_path)
        try:
            onnx.checker.check_model(onnx_model)
            print("onnx model check ok!")
        except onnx.checker.ValidationError as e:
            print(f"model is invalid: {e}")
    
    def io_info(self):
        # 查看模型的输入输出信息
        input_name_list = [input.name for input in self.ort_session.get_inputs()]
        output_name_list = [output.name for output in self.ort_session.get_outputs()]
        print(input_name_list, output_name_list)


if __name__ == "__main__":
    onnx_path = "/data/app/yangyahe/boundary-detector-for-text-correction/checkpoints/onnx/best_batch.onnx"
    ort_infer = OrtInference(onnx_path)
    
    ort_infer.check_onnx()
    ort_infer.io_info()
    
    question = '不忘初心、牢记使命'
    context = "务必不忘初心、牢记使命”，饱含厚重历史感、鲜明时代感、庄严使命感，明确了为中国人民谋幸福、为中华民族谋复兴的初心使命是中国共产党人的永恒追求，凸显了坚守初心使命的时代要求，体现了我们党时刻保持解决大党独有难题的清醒和坚定。"
    ll = len(context)
    
    steps = 1
    ort_inputs = ort_infer.dataset(question, context)
    start = time.time()
    for i in tqdm(range(steps)):
        # label = ort_infer.inference(sample_text)[0]
        start_end_pos = ort_infer.inference_only(ort_inputs)
        if len(start_end_pos) >= 2 and start_end_pos[0] < start_end_pos[1] and start_end_pos[1] <= ll:
            print(context[start_end_pos[0]:start_end_pos[1]])
        else:
            print("没有发现时政长词错误")
    print('onnxruntime(不含构造数据) 平均耗时: ', (time.time() - start) * 1000 / steps, ' ms')
