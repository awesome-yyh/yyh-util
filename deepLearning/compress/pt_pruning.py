import json
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertConfig, BertModel


class PruneBert():
    def __init__(self, origin_model, origin_config, add_pooling_layer=True, layer_ids=range(12)):
        self.layer_ids = layer_ids
        self.origin_model = origin_model
        self.model_paramerts = self.named_parameters()
        self.config = origin_config
        self.config["num_hidden_layers"] = len(self.layer_ids)  # 修改配置文件
    
    def named_parameters(self,):
        # 提取我们想要的层的权重并重命名
        cur_layer_id = 0
        origin_layer_id_last = 0
        model_paramerts = {}
        for name, param in self.origin_model.named_parameters():
            name_split = name.split('.')
            if name.startswith('embeddings') or name.startswith('pooler.layer'):
                model_paramerts[name] = param
            elif name.startswith('encoder.layer') and int(name_split[2]) in self.layer_ids:
                if int(name_split[2]) != origin_layer_id_last:
                    cur_layer_id += 1
                    origin_layer_id_last = int(name_split[2])
                model_paramerts['encoder.layer.' + str(cur_layer_id) + '.' + '.'.join(name_split[3:])] = param
        return model_paramerts
    
    def state_dict(self):
        prune_model = self.origin_model.state_dict()
        for name in list(prune_model.keys()):
            if name in self.model_paramerts:
                prune_model[name] = self.model_paramerts[name]
            else:
                del prune_model[name]
        return prune_model
    
    def print_model(self):
        for name, param in self.origin_model.named_parameters():
            print(name, "==", param.shape)
        print("=" * 20 + ">")
        for name, param in self.named_parameters().items():
            print(name, "==", param.shape)
    
    def save(self, model_save_path):
        torch.save(self.state_dict(), f"{model_save_path}/pytorch_model.bin")
        with open(f"{model_save_path}/config.json", 'w') as fp:
            fp.write(json.dumps(self.config))
        with open(f"{model_save_path}/vocab.txt", 'w') as fp:
            fp.write(open(f"{MODEL_PATH}/vocab.txt").read())
        print("saved!")


if __name__ == "__main__":
    MODEL_PATH = "/Users/yaheyang/.cache/huggingface/hub/models--bert-base-chinese/snapshots/84b432f646e4047ce1b5db001d43a348cd3f6bd0"
    config_file = f"{MODEL_PATH}/config.json"
    
    origin_config = json.loads(open(config_file, 'r').read())
    origin_model = BertModel.from_pretrained(MODEL_PATH)
    
    bp = PruneBert(origin_model, origin_config, layer_ids=range(7))
    bp.print_model()
    print('=' * 20)
    print(bp.config)
    
    model_save_path = "./prune-bert-base-chinese"
    bp.save(model_save_path)
    
