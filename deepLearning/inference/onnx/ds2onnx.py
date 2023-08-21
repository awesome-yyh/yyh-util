import os
from pathlib import Path
import sys
import torch
import torch.onnx
import onnx
from transformers import BertTokenizer
import yaml

sys.path.append(str(Path(__file__).parent.parent.absolute()))
from model import MultiClassLabelModel


class DS2Onnx():
    def __init__(self, input_model, output_model, device_id=0, config_file='configs/train.yaml') -> None:
        with open(config_file, 'r') as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        
        self.input_model = input_model
        self.output_model = output_model
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    
    def load_model(self, ):
        self.model = MultiClassLabelModel(base_model_path=self.cfg["base_model_path"], label_num=self.cfg["label_num"])
        
        self.model.load_state_dict(torch.load(self.input_model, map_location=self.device), strict=True)
        
        self.model.to(self.device)
        self.model.eval()
    
    def dataset(self, ):
        tokenizer = BertTokenizer.from_pretrained(self.cfg["base_token_path"])
        
        sample_text = "北京币石景山区石景山学校"
        
        inputs = tokenizer(sample_text, add_special_tokens=True, padding='max_length', truncation=True, max_length=self.cfg["max_len"], return_tensors='pt')
        
        dummy_input = tuple(value.to(self.device) for value in inputs.values())
        
        input_name_list = list(inputs.keys())
        print("input_name_list: ", input_name_list)
        
        return dummy_input, input_name_list
    
    def check_onnx(self, ):
        onnx_model = onnx.load(self.output_model)
        try:
            onnx.checker.check_model(onnx_model)
            print("onnx model check ok!")
        except onnx.checker.ValidationError as e:
            print(f"model is invalid: {e}")
    
    def __call__(self, ):
        dir_name = os.path.dirname(self.output_model)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)

        self.load_model()
        
        dummy_input, input_name_list = self.dataset()

        output_name_list = ["labels"]
        
        torch.onnx.export(self.model,
                          dummy_input,
                          self.output_model,
                          export_params=True,
                          opset_version=self.cfg["onnx_version"],
                          do_constant_folding=True,
                          input_names=input_name_list,
                          output_names=output_name_list,
                          dynamic_axes={input_name: {0: "batch", 1: "seq"} for input_name in input_name_list},
                          verbose=False)
        print('模型已转为ONNX==>', self.output_model)
        self.check_onnx()


if __name__ == "__main__":
    input_model = "xxx.pt"
    output_model = "api_model/onnx/2/best_model.onnx"
    ds2onnx = DS2Onnx(input_model, output_model)
    ds2onnx()
