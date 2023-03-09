import os
import sys
import hydra
import torch
import torch.onnx
from transformers import BertTokenizer
from pathlib import Path
sys.path.append(str(Path(__file__).absolute()))
from model import MultiClassLabelModel


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def torch2onnx(cfg):
    cuda_condition = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_condition else "cpu")
    
    model = MultiClassLabelModel(base_model_path=cfg.base_model_path, label_num=cfg.label_num)
    model.load_state_dict(torch.load(cfg.best_model_path, map_location=device))
    model.to(device)
    model.eval()
    
    tokenizer = BertTokenizer.from_pretrained(cfg.base_model_path)
    sample_text = "北京币石景山区石景山学校"
    token_dict = tokenizer(
        sample_text,  # [cls]句子1[sep]句子2[sep]
        padding='max_length',  # 不足时填充到指定最大长度
        truncation=True,  # 过长时截断
        max_length=cfg.max_len,  # 2个句子加起来的长度
        return_tensors='pt')  # 返回字典, input_ids, token_type_ids, attention_mask
    dummy_input = tuple(value.to(device) for value in token_dict.values())
    
    try:
        os.makedirs(cfg.onnx_dir)
    except OSError as ex:
        pass  # ignore existing dir
    onnx_out_path = os.path.join(cfg.onnx_dir, 'dynamic_model.onnx')
    
    input_name_list = ["input_ids", "token_type_ids", "attention_mask"]
    output_name_list = ["label"]
    
    torch.onnx.export(model,
                      dummy_input,
                      onnx_out_path,
                      export_params=True,
                      opset_version=cfg.onnx_version,
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
