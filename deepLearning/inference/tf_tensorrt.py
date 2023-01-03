from tensorflow.python.compiler.tensorrt import trt_convert as trt


# 转换成tensorrt图
pb_file_path = "models/multiModel/textcnn/1"
# 优化模型
params=trt.DEFAULT_TRT_CONVERSION_PARAMS
params._replace(precision_mode=trt.TrtPrecisionMode.FP32)
converter = trt.TrtGraphConverterV2(input_saved_model_dir=pb_file_path,
                                    conversion_params=params)
converter.convert()#完成转换,但是此时没有进行优化,优化在执行推理时完成
converter.save(pb_file_path)
