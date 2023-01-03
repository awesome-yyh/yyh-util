import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # 自动随机切分训练数据和测试数据
import torch
from torch.utils.data import Dataset


# pytorch的基本信息
print("------pytorch的基本信息-------")
print("pytorch version: ", torch.__version__)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("mps" if torch.backends.mps.is_available() else 'cpu') # mac m1 gpu: mps
print("CPU or GPU: ", device)


# 创建数据"
print("------创建数据-------")
print(type(torch.tensor([[[1,2,3],[3,4,5]]])))
print(torch.ones((2,3,4)).shape)
print(torch.zeros((2,3,4)).numpy())
print(torch.arange(12).reshape(3,4))

print(torch.ones((3, 2, 1), device='mps')) # 直接在gpu上创建(比在CPU创建后移动到 GPU 上快很多)
print(torch.ones((3, 2, 1)).to('mps'))


# 四则运算
print("------四则运算-------")
A = torch.tensor([[1.0, 2.0], 
                 [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0], 
                 [7.0, 8.0]])

print(torch.add(A, B)) # [[6,8],[10,12]] 对应位相加
print(A+B) # 同上, [[6,8],[10,12]] 对应位相加

print(torch.matmul(A, B)) # [[19,22],[43,50]] 矩阵乘法，对应位相乘并相加
print(A @ B) # 同上，[[19,22],[43,50]] 矩阵乘法，对应位相乘并相加

print(A * B) # [[5,12],[21,32]] 对应位置相乘


# 元素访问和修改
print("------元素访问和修改-------")
T = torch.tensor([[1.0, 2.0], 
                 [3.0, 4.0]])
print(T[-1], T[0:1], T[0,1].item())
T[0,1] = 99 # 原地操作（在原内存地址修改并生效）
print(T)


# 统计运算
print("------统计运算-------")
print(torch.max(A)) # 4, 最大值
print(torch.argmax(A)) # 3, 最大值的索引
print(torch.mean(A)) # 2.5 所有元素的平均值
print(torch.mean(A, axis=1)) # [1.5 3.5]


# 求导
print("------求导-------")
x = torch.tensor([3.0], requires_grad=True)
y = x ** 2
y.backward() #反向传播,求解导数
print("x.grad = ", x.grad) # 6


# 模型演示-线性回归
print("------模型演示-线性回归-------")
torch.set_default_tensor_type(torch.DoubleTensor)
# 读取数据
# y = 2*x - 1
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0], dtype=float)
xs = xs.reshape(xs.shape[0], 1)
ys = ys.reshape(ys.shape[0], 1)

# 探索分析
# plt.plot(xs, ys, 'ro', label='Original data')
# plt.show()

# 数据清洗(缺失值、重复值、异常值、大小写、标点)

# 数据采样(搜集、合成、过采样、欠采样、阈值移动、loss加权、评价指标)

# 特征工程(数值、类别、时间、文本、图像)

# 划分训练集和测试集, random_state是随机数的种子，不填则每次都不同
train_x, test_x, train_y, test_y = train_test_split(xs, ys, test_size=0.2, random_state=1)
print(train_x, test_x, train_y, test_y)


# 搭建模型
class Linear(torch.nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.linear = torch.nn.Linear(1,1) # 包括两个参数是weight和bias
        
    def forward(self,x): # 直接调用对象时, 会自动将传入的参数传到forward函数当中进行计算
        y_pred = self.linear(x)
        return y_pred

model = Linear()

# 查看模型结构
print(model)
print(list(model.parameters()))

criterion = torch.nn.MSELoss(reduction='mean') # loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # # 优化器对象创建时需要传入参数，这里的参数取得是模型对象当中的w和bias

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    train_x = torch.tensor(train_x) # 不共享内存
    train_y = torch.tensor(train_y)
    # train_x = torch.from_numpy(train_x) # 共享内存
    # train_y = torch.from_numpy(train_y)
    
    y_pred = model(train_x) # 前馈得到估计值
    loss = criterion(y_pred, train_y) # 计算损失
    
    optimizer.zero_grad() # 先进行梯度归零
    loss.backward() # 进行反向传播
    optimizer.step() # 进行权重更新
    
    if (epoch+1) % 5 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))


# 经过100次迭代之后输出权重和偏置
print("Weight=",model.linear.weight.item())
print("Bias=",model.linear.bias.item())

# # Plot the graph
# predicted = model(torch.tensor(train_x)).detach().numpy()
# plt.plot(train_x, train_y, 'ro', label='Original data')
# plt.plot(train_x, predicted, label='Fitted line')
# plt.legend()
# plt.show()

# torch.save(model.state_dict(), 'model.ckpt')

import torch.onnx 

#Function to Convert to ONNX 
def Convert_ONNX(): 

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    input_size = 1
    dummy_input = torch.randn(1, input_size, requires_grad=True)  

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "HelloPytorch.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX')

Convert_ONNX()
