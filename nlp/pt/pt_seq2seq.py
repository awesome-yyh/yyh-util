from transformers import BertTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM


base_model = "/data/app/base_model/shibing624-bart4csc-base-chinese"  # 文本纠错
base_model = '/data/app/base_model/csebuetnlp-mT5_multilingual_XLSum'  # 摘要
base_model = '/data/app/base_model/Helsinki-NLP-opus-mt-zh-en'  # 翻译-中到英
base_model = '/data/app/base_model/Helsinki-NLP-opus-mt-en-zh'  # 翻译-英到中

# 大模型，使用 AutoModelForCausalLM
base_model = '/data/app/base_model/bigscience-bloomz-7b1'
base_model = '/data/app/base_model/bigscience-bloom-560m'
base_model = '/data/app/base_model/bigscience-bloom-7b1'
base_model = '/data/app/base_model/BelleGroup-BELLE-7B-2M'

base_model = '/data/app/base_model/google-flan-t5-base'
base_model = '/data/app/base_model/google-flan-t5-large'
base_model = '/data/app/base_model/google-mt5-base'
base_model = '/data/app/base_model/google-mt5-large'
base_model = '/data/app/base_model/imxly-t5-pegasus'
base_model = '/data/app/base_model/ClueAI-ChatYuan-large-v2'
base_model = '/data/app/base_model/csebuetnlp-mT5_multilingual_XLSum'  # mt5 base

base_model = '/data/app/base_model/shibing624-mengzi-t5-base-chinese-correction'  # 2.5亿
base_model = '/data/app/base_model/pszemraj-flan-t5-large-grammar-synthesis'  # 7.8亿

tokenizer = AutoTokenizer.from_pretrained(base_model)
# model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")  # load_in_8bit=True
model = AutoModelForSeq2SeqLM.from_pretrained(base_model, device_map="auto")  # t5用

print("Total Parameters:", sum([p.nelement() for p in model.parameters()]) / 1e8, "亿")

input_text = 'how are you'
input_text = "Answer the following question by reasoning step-by-step.What is the boiling point of Nitrogen?"  # flan t5的prompt
input_text = "translate English to German: That is good"  # t5的prompt

input_text = "你是一个出色的文本校对编辑，请对这句话进行纠错：又开始沙尘暴了，外面灰兔兔的，空气中迷漫着各种汇晨和雾霾"
input_text = "又开始沙尘暴了，外面灰兔兔的，空气中迷漫着各种汇晨和雾霾"
input_text = "一周有几天，答案："
input_text = "“翻译成英文：\n时代的尘埃，落在昔通人身上就是一座大山\n答案："  # chatyuan的prompt

input_ids = tokenizer.encode(input_text, add_special_tokens=True, return_tensors="pt").to("cuda")

# outputs = model(input_ids)

generated_ids = model.generate(input_ids, max_length=50, no_repeat_ngram_size=3, num_beams=4)
# no_repeat_ngram_size  防止模型生成过多的相同的片段
# max_length  表示生成的文本的最大长度
# do_sample=True  意味着使用基于采样的方法生成文本
# temperature  控制生成文本的多样性和保守度

output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(output_text)
