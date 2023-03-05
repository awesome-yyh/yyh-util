import torch


def Convert_ONNX(model): 
    model.eval() # 设置模型为推理模式

    # 创建dummy input tensor  
    input_size = 1
    dummy_input = torch.randn(1, input_size, requires_grad=True)  
    
    # 转换onnx
    torch.onnx.export(model,
        dummy_input,       # 如果有多个输入则使用元组
        "HelloPytorch.onnx",       # onnx保存路径
        export_params=True,  # 将训练好的参数权重存储在模型文件中
        opset_version=10,    # ONNX 版本
        do_constant_folding=True,  # 是否执行常数折叠优化
        input_names = ['modelInput'],   # 输入名
        output_names = ['modelOutput'], # 输出名
        dynamic_axes={ 'modelInput' : {0 : 'batch_size'}, # 指定输入输出张量的哪些维度是动态的
                        'modelOutput': {0 : 'batch_size'}}) 
    print('模型已转为ONNX')
