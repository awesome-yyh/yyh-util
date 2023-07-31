import os
import torch
import torch.onnx
import onnx
from transformers import BertTokenizer, BertForQuestionAnswering
import yaml
# import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).absolute()))
# from model import BoundaryDetectorModel


class DS2Onnx():
    def __init__(self, input_model, output_model, config_file='./configs/train.yaml') -> None:
        self.input_model = input_model
        self.output_model = output_model
        with open(config_file, 'r') as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cfg["CUDA_VISIBLE_DEVICES"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def load_model(self, ):
        self.model = BertForQuestionAnswering.from_pretrained(self.cfg["base_model_path"])
        self.model.load_state_dict(torch.load(self.input_model, map_location=self.device)["module"], strict=True)
        
        self.model.to(self.device)
        self.model.eval()
    
    def model_input(self, ):
        tokenizer = BertTokenizer.from_pretrained(self.cfg["base_model_path"])
        question = "庆祝建团百年"
        context = "5月7日上午，常州港航团总支深入基层，开展“建团百路青春正当时”主题志愿服务活动，以庆祝建团，展示青年风采，弘扬五四精神。"
        
        inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, max_length=self.cfg["max_len"], truncation="only_second", padding='max_length', return_tensors='pt')
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        token_type_ids = inputs['token_type_ids'].to(self.device)
        dummy_input = (input_ids, attention_mask, token_type_ids)
        input_name_list = ["input_ids", "attention_mask", "token_type_ids"]
        
        return dummy_input, input_name_list
    
    def __call__(self, ):
        dir_name = os.path.dirname(self.output_model)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)

        self.load_model()
        
        dummy_input, input_name_list = self.model_input()

        output_name_list = ["start_pos", "end_pos"]
        
        torch.onnx.export(self.model,
                          dummy_input,
                          self.output_model,
                          export_params=True,
                          opset_version=self.cfg["onnx_version"],
                          do_constant_folding=True,
                          input_names=input_name_list,
                          output_names=output_name_list,
                          dynamic_axes={
                              input_name_list[0]: {0: "batch", 1: "seq"}, input_name_list[1]: {0: "batch", 1: "seq"}, input_name_list[2]: {0: "batch", 1: "seq"}},
                          verbose=False)
        print('模型已转为ONNX==>', self.output_model)
        
    def check_onnx(self, ):
        onnx_model = onnx.load(self.output_model)
        try:
            onnx.checker.check_model(onnx_model)
            print("onnx model check ok!")
        except onnx.checker.ValidationError as e:
            print(f"model is invalid: {e}")


if __name__ == "__main__":
    input_model = "/data/app/yangyahe/boundary-detector-for-text-correction/checkpoints/best/mp_rank_00_model_states.pt"
    
    output_model = "/data/app/yangyahe/boundary-detector-for-text-correction/checkpoints/onnx/best_batch.onnx"
    ds2onnx = DS2Onnx(input_model, output_model)
    ds2onnx()
    ds2onnx.check_onnx()
