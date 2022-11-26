import torch


# pytorch的基本信息
print("------pytorch的基本信息-------")
print("pytorch version: ", torch.__version__)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device=torch.device("mps") # mac m1 gpu: mps
print("CPU or GPU: ", device)

print("# 创建数据")
x = torch.arange(12)
print(x)
print(x.shape)
x = x.reshape(3,4)
print(x)
print(x.shape)
print(torch.zeros((2,3,4)))
print(torch.ones((2,3,4)))
print(torch.tensor([[[1,2,3],[3,4,5]]]).shape)

print("# numpy数据转换")
n = x.numpy()
t = torch.tensor(n)
print(type(n), type(t))

print("# 元素访问和修改")
print(x[-1], x[1:3], x[1,2].item())
x[1,2] = 99 # 原地操作（在原内存地址修改并生效）
print(x)

print("# 算术运算")
x = torch.tensor([1.0,2,4])
y = torch.tensor([3,5,7])
print(x+y, x-y, x*y, x/y, x**y, torch.exp(x)) # 按元素进行,形状不同也可操作（广播机制）

print("# 统计运算")
print(x.sum())

print("# 张量拼接")
print(torch.cat((x.reshape(1,3),y.reshape(1,3)), dim=0)) # 在第0维上合并，纵向拼接
print(torch.cat((x.reshape(1,3),y.reshape(1,3)), dim=1)) # 在第1维上合并，横向拼接

