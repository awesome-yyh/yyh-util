
# bert
print("bert")
import tokenization, modeling
import tensorflow as tf
import os, json
import numpy as np

if(tf.__version__.startswith("2")):
    tf.gfile = tf.io.gfile
    tf.flags = tf.compat.v1.flags
flags = tf.flags
FLAGS = flags.FLAGS

# 设置bert_config_file
bert_path = "./models/chinese_L-12_H-768_A-12/"
flags.DEFINE_string(
    "bert_config_file", os.path.join(bert_path, 'bert_config.json'),
    "The config json file corresponding to the pre-trained BERT model."
)
flags.DEFINE_string(
    'bert_vocab_file', os.path.join(bert_path,'vocab.txt'),
    'the config vocab file',
)
flags.DEFINE_string(
    'init_checkpoint', os.path.join(bert_path,'bert_model.ckpt'),
    'from a pre-trained BERT get an initial checkpoint',
)
flags.DEFINE_bool(
    "use_one_hot_embeddings", False,
    "If True, tf.one_hot will be used for embedding lookups, otherwise "
    "tf.nn.embedding_lookup will be used. On TPUs, this should be True "
    "since it is much faster.")

def convert2Uni(text):
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode('utf-8','ignore')
    else:
        print(type(text))
        print('####################wrong################')

def inputs(vectors, maxlen = 50):
    length = len(vectors)
    if length > maxlen:
        return vectors[0:maxlen], [1]*maxlen, [0]*maxlen
    else:
        input = vectors+[0]*(maxlen-length)
        mask = [1]*length + [0]*(maxlen-length)
        segment = [0]*maxlen
        return input, mask, segment

def response_request(text):
    vectors = [dictionary.get('[CLS]')] + [dictionary.get(i) if i in dictionary else dictionary.get('[UNK]') for i in list(text)] + [dictionary.get('[SEP]')]
    input, mask, segment = inputs(vectors)

    input_ids = np.reshape(np.array(input), [1, -1])
    input_mask = np.reshape(np.array(mask), [1, -1])
    segment_ids = np.reshape(np.array(segment), [1, -1])

    embedding = tf.squeeze(ftModel.get_sequence_output())
    rst = sess.run(embedding, feed_dict={'input_ids_p:0':input_ids, 'input_mask_p:0':input_mask, 'segment_ids_p:0':segment_ids})

    return json.dumps(rst.tolist(), ensure_ascii=False)

dictionary = tokenization.load_vocab(FLAGS.bert_vocab_file)
init_checkpoint = FLAGS.init_checkpoint

sess = tf.compat.v1.Session()

bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

tf.compat.v1.disable_v2_behavior()
input_ids_p = tf.compat.v1.placeholder(shape=[None, None], dtype = tf.int32, name='input_ids_p')
input_mask_p = tf.compat.v1.placeholder(shape=[None, None], dtype = tf.int32, name='input_mask_p')
segment_ids_p = tf.compat.v1.placeholder(shape=[None, None], dtype = tf.int32, name='segment_ids_p')

ftModel = modeling.BertModel(
    config = bert_config,
    is_training = False,
    input_ids = input_ids_p,
    input_mask = input_mask_p,
    token_type_ids = segment_ids_p,
    use_one_hot_embeddings = FLAGS.use_one_hot_embeddings,
)
print('####################################')
restore_saver = tf.train.Saver()
restore_saver.restore(sess, init_checkpoint)

print(response_request('我叫水奈樾。'))
1/0
# 获取对应的embedding 输入数据[batch_size, seq_length, embedding_size]
embedding = ftModel.get_sequence_output()




flags.DEFINE_string(
    "vocab_file",
    "./models/chinese_L-12_H-768_A-12/vocab.txt",
    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 12, 
    "Maximum sequence length.")

flags.DEFINE_string(
    "init_checkpoint", './models/chinese_L-12_H-768_A-12/bert_model.ckpt',
    "Initial checkpoint (usually from a pre-trained BERT model).")

tvars = tf.compat.v1.trainable_variables()
(assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, FLAGS.init_checkpoint)
tf.compat.v1.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map) 

tf.compat.v1.logging.info("**** Trainable Variables ****")
print(tvars)
print(initialized_variable_names)
for var in tvars:
    init_string = ""
    if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
    tf.compat.v1.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                    init_string)
1/0
tokenizer = tokenization.FullTokenizer(
    vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)


1/0

tokens = tokenizer.tokenize(text)
print(tokens)

# Account for [CLS] and [SEP] with "- 2"
if len(tokens) > FLAGS.max_seq_length - 2:
    tokens = tokens[:FLAGS.max_seq_length - 2]

tokens = ["[CLS]"]
tokens.extend(tokens)
tokens.append("[SEP]")
print(tokens)

input_ids = tokenizer.convert_tokens_to_ids(tokens)

while len(input_ids) < FLAGS.max_seq_length:
    input_ids.append(0)

print(input_ids)
