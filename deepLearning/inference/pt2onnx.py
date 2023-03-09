import os
import sys
import hydra
import torch
import torch.onnx
from transformers import BertTokenizer
from pathlib import Path
sys.path.append(str(Path(__file__).absolute()))
from model import XxxlModel


def torch2onnx(base_model_path, best_model_path, max_len, onnx_dir):
    cuda_condition = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_condition else "cpu")
    
    model = XxxlModel()
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()
    
    tokenizer = BertTokenizer.from_pretrained(base_model_path)
    sample_text = "北京币石景山区石景山学校"
    token_dict = tokenizer(
        sample_text,  # [cls]句子1[sep]句子2[sep]
        padding='max_length',  # 不足时填充到指定最大长度
        truncation=True,  # 过长时截断
        max_length=max_len,  # 2个句子加起来的长度
        return_tensors='pt')  # 返回字典, input_ids, token_type_ids, attention_mask
    dummy_input = tuple(value.to(device) for value in token_dict.values())
    
    try:
        os.makedirs(onnx_dir)
    except OSError as ex:
        pass  # ignore existing dir
    onnx_out_path = os.path.join(onnx_dir, 'dynamic_model.onnx')
    
    input_name_list = ["input_ids", "token_type_ids", "attention_mask"]
    output_name_list = ["label"]
    
    torch.onnx.export(model,
                      dummy_input,
                      onnx_out_path,
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True,
                      input_names=input_name_list,
                      output_names=output_name_list,
                      dynamic_axes={input_name_list[0]: [0, 1],
                                    input_name_list[1]: [0, 1],
                                    input_name_list[2]: [0, 1]},
                      verbose=False)
    print('模型已转为ONNX')


if __name__ == "__main__":
    torch2onnx()
