import numpy as np
import torch


# pytorch的基本信息
print("------pytorch的基本信息-------")
print("pytorch version: ", torch.__version__)

device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = "mps"
print("CPU or GPU: ", device)

# print("cuda是否可用: ", torch.cuda.is_available())
# print("cuda版本: ", torch.version.cuda)
# print("cudnn版本: ", torch.backends.cudnn.version())
# print("GPU数量: ", torch.cuda.device_count())
# print("GPU0名: ", torch.cuda.get_device_name(0))  # 设备索引默认从0开始
# print("返回当前设备索引: ", torch.cuda.current_device())


# 创建数据及形状修改
print("------创建数据、数据类型及形状修改-------")
print(type(torch.tensor([[[1, 2, 3], [3, 4, 5]]])))
print(torch.ones((2, 3, 4)).shape)
print(torch.arange(12).reshape(3, 4))  # 修改形状
print(torch.arange(12))
print(torch.arange(12).unsqueeze(0))  # 在哪个地方加一个维度, 0是在最外面套括号
print(torch.arange(12).unsqueeze(1))  # 第1维的每个元素加括号

print(torch.arange(12).unsqueeze(0).squeeze(0))  # 去掉哪个一个维度，0是最外面的括号
print(torch.arange(12).unsqueeze(0).squeeze(1))  # 如果这个维度的元素大于1则不做处理

print(torch.arange(12).reshape(3, 4).permute(1, 0))  # 维度从m*n变成n*m, 对于二维相当于转置

q = torch.tensor([1.0, 3.0], dtype=torch.float32)
a = torch.FloatTensor([1.0, 3.0])  # 和上面的等价
print(a.dtype)  # torch.float32


# 和numpy协同
print(torch.zeros((2, 3, 4)).numpy())
x = np.array([[1, 2, 3], [4, 5, 6]])
print(torch.tensor(x))  # 不共享内存
print(torch.from_numpy(x))  # 共享内存


# 数据转到gpu
tgpu = torch.ones((3, 2, 1)).to('mps')
print(tgpu.cpu())
tgpu = torch.ones((3, 2, 1), device='mps')  # 直接在gpu上创建(比在CPU创建后移动到 GPU 上快很多)
print(tgpu.cpu().numpy())  # gpu上的数先转到cpu后才能转numpy, numpy不支持gpu


# 元素访问和修改
print("------元素的索引访问和修改-------")
x = torch.tensor([[1.0, 2.0],
                 [3.0, 4.0]])
print(x[-1], x[0:1], x[0, 1].item())
x[0, 1] = 99  # 原地操作（在原内存地址修改并生效）
print(x)


# 四则运算
print("------四则运算-------")
A = torch.tensor([[1.0, 2.0],
                 [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0],
                 [7.0, 8.0]])

print(torch.add(A, B))  # [[6,8],[10,12]] 对应位相加
print(A + B)  # 同上, [[6,8],[10,12]] 对应位相加

print(torch.matmul(A, B))  # [[19,22],[43,50]] 矩阵乘法，对应位相乘并相加
print(A @ B)  # 同上，[[19,22],[43,50]] 矩阵乘法，对应位相乘并相加

print(torch.mul(A, B))  # [[5,12],[21,32]] 对应位置相乘
print(A * B)  # 同上, [[5,12],[21,32]] 对应位置相乘


# 统计运算
print("------统计运算-------")
print(torch.max(A))  # 4, 最大值
print(torch.argmax(A))  # 3, 最大值的索引
print(torch.mean(A))  # 2.5 所有元素的平均值
print(torch.mean(A, axis=1))  # [1.5 3.5]


# tensor拼接(同维度拼接)
print("------tensor拼接-------")
x = torch.tensor([[1.0, 2.0], 
                 [3.0, 4.0]])
y = torch.tensor([[5.0, 6.0], 
                 [7.0, 8.0]])
z0 = torch.cat((x, y), dim=0)
z1 = torch.cat((x, y), dim=1)
print(z0)
print(z1)

# tensor stack(扩张维度后再拼接)
# 在自然语言处理和卷及神经网络中， 通常为了保留–[序列(先后)信息] 和 [张量的矩阵信息] 才会使用stack。
print("------tensor stack-------")
x = torch.tensor([[1.0, 2.0], 
                 [3.0, 4.0]])
y = torch.tensor([[5.0, 6.0], 
                 [7.0, 8.0]])
z0 = torch.stack((x, y), dim=0)
z1 = torch.stack((x, y), dim=1)
print(z0)
print(z1)


# 模型的参数初始化
print("------模型的参数初始化-------")
# 返回填充有未初始化数据的张量
w0 = torch.empty(2, 3)
print(w0)

# 将一个不可训练的tensor转换成可以训练的类型parameter
w = torch.nn.Parameter(w0)
print(w)
print(type(w))

# 让数据变为正态分布
z0 = torch.nn.init.normal_(w, mean=0, std=1)
zx = torch.nn.init.xavier_normal_(w)
zkm = torch.nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
print(z0)
print(zx)
print(zkm)


# 求导
print("------求导-------")
x = torch.tensor([3.0], requires_grad=True)
# x.requires_grad = True # 或事后添加需要求导标识 
y = x ** 3
y.backward(retain_graph=True)  # 反向传播,求解导数, retain_graph保持图即更新图中的值下次继续使用
print("关于x的梯度: ", x.grad)  # 27 3*x^2 = 3*3^2 = 27
y.backward(retain_graph=True)
print("关于x的梯度: ", x.grad)  # 54 = 27+27在第二次反向传播时，将自动和第一次的梯度相加
y.backward(retain_graph=True)
print("关于x的梯度: ", x.grad)  # 81 = 54+27, 会继续累加

x.grad.data.zero_()  # 梯度清零
y.backward()
print("关于x的梯度: ", x.grad)  # 27, 梯度清零后再求导又得到27
