import tensorrt as trt


logger = trt.Logger(trt.Logger.WARNING)  # 要创建构建器，需要首先创建一个记录器
builder = trt.Builder(logger)  # 创建一个构建器

network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))  # 创建网络定义, 为了使用 ONNX 解析器导入模型，需要EXPLICIT_BATCH标志

parser = trt.OnnxParser(network, logger)  # 创建一个 ONNX 解析器来填充网络定义

model_path = "checkpoints/onnx/model.onnx"
success = parser.parse_from_file(model_path)  # 读取模型文件并处理任何错误
for idx in range(parser.num_errors):
    print(parser.get_error(idx))


if not success:
    pass  # Error handling code here

# profile = builder.create_optimization_profile()
# # INPUT0可以接收[1, 2] -> [max_batch, 2]的维度
# max_batch = 64
# profile.set_shape("INPUT0", [1, 2], [1, 2], [max_batch, 2])
# profile.set_shape("INPUT1", [1, 2], [1, 2], [max_batch, 2])

config = builder.create_builder_config()  # 指定 TensorRT 应该如何优化模型
# config.add_optimization_profile(profile)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1<<30: 1KB, 1<<20: MiB, 1<<30: 1GB

try:
    engine_bytes = builder.build_serialized_network(network, config)
except AttributeError:
    engine = builder.build_engine(network, config)
    engine_bytes = engine.serialize()
    del engine

# serialized_engine = builder.build_serialized_network(network, config)  # 构建和序列化engine


with open("sample.engine", "wb") as f:  # 保存engine文件
    f.write(engine_bytes)
