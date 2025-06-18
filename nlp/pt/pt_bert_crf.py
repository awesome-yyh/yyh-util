import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torchcrf import CRF

class BERT_LSTM_CRF(nn.Module):
    def __init__(self, bert_model_name, hidden_dim, num_labels):
        super(BERT_LSTM_CRF, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(input_size=self.bert.config.hidden_size,
                            hidden_size=hidden_dim,
                            num_layers=1,
                            bidirectional=True,
                            batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        # BERT embeddings
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]

        # LSTM layer
        lstm_out, _ = self.lstm(sequence_output)

        # Linear layer to get emissions
        emissions = self.hidden2tag(lstm_out)

        if labels is not None:
            # CRF loss
            loss = -self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')
            return loss
        else:
            # CRF decode
            prediction = self.crf.decode(emissions, mask=attention_mask.byte())
            return prediction

# Define the model
bert_model_name = '/data/app/yangyahe/base_model/google-bert-bert-base-uncased'
hidden_dim = 128
num_labels = 9  # Number of NER tags

model = BERT_LSTM_CRF(bert_model_name, hidden_dim, num_labels)

# Define a tokenizer
tokenizer = BertTokenizer.from_pretrained(bert_model_name)

# Example sentences and labels
sentences = ["Hello, my name is John Doe and I live in New York."]
labels = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 0]]  # Example labels

# Tokenize input sentences
inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, is_split_into_words=False)
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
token_type_ids = inputs['token_type_ids']

# Pad labels to match input_ids length
max_len = input_ids.shape[1]
padded_labels = [label + [0] * (max_len - len(label)) for label in labels]
labels_tensor = torch.tensor(padded_labels)

# Print shapes to ensure they match
print("Input IDs shape:", input_ids.shape)
print("Labels shape:", labels_tensor.shape)
# Forward pass with labels (for training)
loss = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels_tensor)
print("Loss:", loss.item())

# Forward pass without labels (for prediction)
predictions = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
print("Predictions:", predictions)