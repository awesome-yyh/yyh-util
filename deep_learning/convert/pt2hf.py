import os
from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM, BertTokenizer, BertModel
from peft import PeftModel, PeftConfig


hf_model = "shibing624-text2vec-base-chinese"

input_model_state = None
# input_model_state = "checkpoints/e4f117_mp_rank_00_model_states.pt"

peft_model_id = None
# peft_model_id = "lora/epoch_29_file_1_end_global_step9720"

output_model_pretrained = Path(input_model_state).parent

# model = AutoModelForSeq2SeqLM.from_pretrained(hf_model)
model = BertModel.from_pretrained(hf_model)

# model.resize_token_embeddings(32596+4)  # len(self.tokenizer))

# 加载参数
if input_model_state:
    model.load_state_dict(torch.load(input_model_state)["module"], strict=True)

# 加LoRA，并合并进原模型
if peft_model_id:
    print(f"加lora: {peft_model_id}")
    model = PeftModel.from_pretrained(model, peft_model_id)
    model = model.merge_and_unload()

# 保存为huggingface格式
model.save_pretrained(output_model_pretrained)  # 指定保存路径
print("模型已保存在: ", output_model_pretrained)

# 之后推理时即可直接使用
# model = AutoModelForSeq2SeqLM.from_pretrained("checkpoints/model.bin")
