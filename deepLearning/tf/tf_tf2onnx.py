import numpy as np
# 转换成onnx
# python -m tf2onnx.convert   --saved-model ~/mypython/yyh-util/models/multiModel/mlp/1 --output ~/xxx.onnx --opset 7

# 查看onnx模型
import onnx
model = onnx.load("/Users/yaheyang/xxx.onnx")
print(model)
# 使用netron（https://netron.app/），图像化显示ONNX模型的计算拓扑图

# 使用onnx模型
import onnxruntime as ort
session = ort.InferenceSession("/Users/yaheyang/xxx.onnx", None)
input_name = session.get_inputs()[0].name
label_name = session.get_outputs()[0].name

test_input = np.expand_dims(test_images[0],0)
outputs = session.run(None, {input_name: test_input.astype(np.float32)})
print(f"第一张图片的预测值: {np.argmax(outputs)}")
print(f"第一张图片的真实值: {test_labels[0]}")

# # 转换成tensorrt
# from tensorflow.python.compiler.tensorrt import trt_convert as trt
# params=trt.DEFAULT_TRT_CONVERSION_PARAMS
# params._replace(precision_mode=trt.TrtPrecisionMode.FP32)
# converter = trt.TrtGraphConverterV2(input_saved_model_dir=pb_file_path,conversion_params=params)
# converter.convert()#完成转换,但是此时没有进行优化,优化在执行推理时完成
# pb_file_path = './models/multiModel/mlptrt/1'
# converter.save(pb_file_path)

# saved_model_loaded = tf.saved_model.load(
#     "trt_savedmodel", tags=[trt.tag_constants.SERVING])#读取模型
# graph_func = saved_model_loaded.signatures[
#     trt.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]#获取推理函数,
# # 也可以使用saved_model_loaded.signatures['serving_default']
# frozen_func = trt.convert_to_constants.convert_variables_to_constants_v2(
#     graph_func)#将模型中的变量变成常量,这一步可以省略,直接调用graph_func也行

# test_input = np.expand_dims(test_images[0],0)
# pred = frozen_func(test_input) # 模型预测
# print(f"第一张图片的预测值: {np.argmax(pred)}")
# print(f"第一张图片的真实值: {test_labels[0]}")
