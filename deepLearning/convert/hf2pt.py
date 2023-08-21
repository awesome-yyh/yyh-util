from collections import namedtuple
import os
import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM, BertTokenizer, BertModel
from peft import PeftModel, PeftConfig


class MyNet(nn.Module):
    def __init__(self, hf_model, input_model_state=None, peft_model_id=None) -> None:
        super(MyNet, self).__init__()
        
        # self.model = AutoModelForSeq2SeqLM.from_pretrained(hf_model)
        self.model = BertModel.from_pretrained(hf_model)

        # self.model.resize_token_embeddings(32596+4)  # len(self.tokenizer))

        # 加载参数
        if input_model_state:
            print(f"加载参数: {input_model_state}")
            self.model.load_state_dict(torch.load(input_model_state)["module"], strict=True)

        # 加LoRA，并合并进原模型
        if peft_model_id:
            print(f"加lora: {peft_model_id}")
            self.model = PeftModel.from_pretrained(self.model, peft_model_id)
            self.model = self.model.merge_and_unload()
    
    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, input_ids, token_type_ids, attention_mask):
        model_output = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        # Perform pooling. In this case, mean pooling.
        sentence_embeddings = self.mean_pooling(model_output[0], attention_mask)
                
        return sentence_embeddings


if __name__ == "__main__":
    hf_model = "shibing624-text2vec-base-chinese"

    input_model_state = None
    # input_model_state = "checkpoints/e4f117_mp_rank_00_model_states.pt"

    peft_model_id = None
    # peft_model_id = "lora/epoch_29_file_1_end_global_step9720"

    my_model = MyNet(hf_model, input_model_state=input_model_state, peft_model_id=peft_model_id)

    # model test
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    my_model = my_model.to(device)
    tokenizer = BertTokenizer.from_pretrained(hf_model)

    sentences = ['今天是星期六', '晴天', '雨天']
    # Tokenize sentences
    encoded_input = tokenizer(sentences, max_length=128, padding=True, truncation=True, return_tensors='pt').to(device)
    # Compute token embeddings
    print(encoded_input, encoded_input['input_ids'].shape)

    with torch.no_grad():
        model_output = my_model(**encoded_input)
    print("sentence_embeddings: ", model_output, type(model_output), model_output.shape)

    # 保存为pytorch格式
    # options = namedtuple("options", encoded_input)
    # print("options: ", options(**encoded_input), type(options(**encoded_input)))

    dummy_inputs = list(encoded_input.values())
    print("dummy_inputs: ", dummy_inputs, encoded_input.keys())
    traced_model = torch.jit.trace(my_model, dummy_inputs)

    output_path = os.path.join(hf_model, 'model.pt')
    traced_model.save(output_path)
    print("model saved to path: ", output_path)
