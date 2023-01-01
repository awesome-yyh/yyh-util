import numpy as np
# 转换成onnx
# python -m tf2onnx.convert   --saved-model ~/mypython/yyh-util/models/multiModel/mlp/1 --output ./xxx.onnx --opset 7

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

# test_input = np.expand_dims(test_images[0],0)
# outputs = session.run(None, {input_name: test_input.astype(np.float32)})
# print(f"第一张图片的预测值: {np.argmax(outputs)}")
# print(f"第一张图片的真实值: {test_labels[0]}")
