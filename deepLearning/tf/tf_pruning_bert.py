import tensorflow as tf
import os


# 对训练好的bert模型进行剪枝，并重新保存

sess = tf.Session()
last_name = 'bert_model.ckpt'
model_path = 'chinese_L-12_H-768_A-12'
imported_meta = tf.train.import_meta_graph(os.path.join(model_path, last_name + '.meta'))
imported_meta.restore(sess, os.path.join(model_path, last_name))
init_op = tf.local_variables_initializer()
sess.run(init_op)

bert_dict = {}
# 获取待保存的层数节点
for var in tf.global_variables():
    # print(var)
    # 提取第0层和第11层和其它的参数，其余1-10层去掉，存储变量名的数值
    if var.name.startswith('bert/encoder/layer_') and not var.name.startswith(
            'bert/encoder/layer_0') and not var.name.startswith('bert/encoder/layer_11'):
        pass
    else:
        bert_dict[var.name] = sess.run(var).tolist()

# print('bert_dict:{}'.format(bert_dict))
# 真实保存的变量信息
need_vars = []
for var in tf.global_variables():
    if var.name.startswith('bert/encoder/layer_') and not var.name.startswith(
            'bert/encoder/layer_0/') and not var.name.startswith('bert/encoder/layer_1/'):
        pass
    elif var.name.startswith('bert/encoder/layer_1/'):
        # 寻找11层的var name，将11层的参数给第一层使用
        new_name = var.name.replace("bert/encoder/layer_1", "bert/encoder/layer_11")
        op = tf.assign(var, bert_dict[new_name])
        sess.run(op)
        need_vars.append(var)
        print(var)
    else:
        need_vars.append(var)
        print('####',var)

# 保存model
saver = tf.train.Saver(need_vars)
saver.save(sess, os.path.join('chinese_L-12_H-768_A-12_pruning', 'bert_pruning_2_layer.ckpt'))
