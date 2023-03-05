from transformers import BertTokenizerFast
from textgen import BartSeq2SeqModel

tokenizer = BertTokenizerFast.from_pretrained('shibing624/bart4csc-base-chinese')
model = BartSeq2SeqModel(
    encoder_type='bart',
    encoder_decoder_type='bart',
    encoder_decoder_name='shibing624/bart4csc-base-chinese',
    tokenizer=tokenizer,
    args={"max_length": 128, "eval_batch_size": 128})
sentences = ["这时亿个文本纠错的安例"]
print(model.predict(sentences))  # ['这是一个文本纠错的案例']
