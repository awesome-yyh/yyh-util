import torch
import torch.nn as nn
from torch import optim
import torchvision as tv
from torchvision.transforms import ToTensor
import argparse
from pt_LeNet import LeNet_5


device = torch.device("mps")
#使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser()
parser.add_argument('--outf', default='./models/', help='folder to output images and model checkpoints') #模型保存路径
parser.add_argument('--net', default='./models/net.pth', help="path to netG (to continue training)")  #模型加载路径
opt = parser.parse_args()

# 超参数设置
EPOCH = 5   #遍历数据集次数
BATCH_SIZE = 64      #批处理尺寸(batch_size)
LR = 0.001        #学习率


# 下载训练集
trainset = tv.datasets.MNIST(
    root='./data/',
    train=True,
    download=True,
    transform=ToTensor()) # transform修改样本，target_transform修改label

# 下载测试集
testset = tv.datasets.MNIST(
    root='./data/',
    train=False,
    download=True,
    transform=ToTensor())

# 使用DataLoader迭代Dataset
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    )

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    )

for X, y in trainloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# 定义损失函数loss function 和优化方式（采用SGD）
net = LeNet_5().to(device)
print(net)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)


for epoch in range(EPOCH):
    sum_loss = 0.0
    # 数据读取
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # 梯度清零
        optimizer.zero_grad()

        # forward + backward
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 每训练100个batch打印一次平均loss
        sum_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %d] loss: %.03f'
                    % (epoch + 1, i + 1, sum_loss / 100))
            sum_loss = 0.0
    # 每跑完一次epoch测试一下准确率
    with torch.no_grad():
        correct = 0
        total = 0
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            # 取得分最高的那个类
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('第%d个epoch的识别准确率为：%d%%' % (epoch + 1, (100 * correct / total)))
torch.save(net.state_dict(), '%s/net_%03d.pth' % (opt.outf, epoch + 1))
