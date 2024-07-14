'''
Author: yyh owyangyahe@126.com
Date: 2023-02-09 20:06:56
LastEditors: yyh owyangyahe@126.com
LastEditTime: 2024-07-04 16:57:41
FilePath: /mypython/yyh-util/deepLearning/pytorch/pt_trainer.py
Description: 
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from pt_moe import MoE
from pt_model import CNN, RNN, MLP, BiRNNModel, BiLSTMModel, BiGRUModel, TransformerModel


device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')


# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载 MNIST 数据集
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)


# 初始化网络和优化器
# net = MLP()  # 96.76%, 53.5818w
# net = CNN()  # 98.38%, 42.1642w
# net = RNN(input_size=28, hidden_size=256, num_layers=4, num_classes=10)  #  92.27%, 47.0538w
net = BiRNNModel(input_dim=28, hidden_dim=256, num_layers=2, num_classes=10)  # 92.62%, 54.5802w
# net = BiLSTMModel(input_dim=28, hidden_dim=128, num_layers=2, num_classes=10)  # 96.61, 55.9626w
# net = BiGRUModel(input_dim=28, hidden_dim=128, num_layers=2, num_classes=10)  # 96.280%, 42.0362w

# net = TransformerModel(input_dim=784, embed_dim=128, num_heads=4, num_layers=2, num_classes=10)  # 95.75% 138.817w
# net = MoE(input_size=784, output_size=10, num_experts=10, hidden_size=128, noisy_gating=True, k=4)  # 94.35, 103.338w

print(net)
print(f"总参数量: {sum([p.nelement() for p in net.parameters()]) / 1e4} w")
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 训练网络
net.train()
for epoch in range(1):  # 训练 epoch
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
                
        optimizer.zero_grad()
        
        # outputs, aux_loss = net(inputs.view(inputs.shape[0], -1))  # MoE
        outputs = net(inputs.squeeze(1))  # RNN
        # outputs = net(inputs)
        
        loss = criterion(outputs, labels)
        # total_loss = loss + aux_loss  # MOE
        total_loss = loss
        total_loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # 每 100 个 mini-batch 打印一次
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')


correct = 0
total = 0
net.eval()
with torch.no_grad():
    for data in testloader:
        images, labels = data
        inputs, labels = images.to(device), labels.to(device)
        
        # outputs, _ = net(inputs.view(inputs.shape[0], -1)) # MOE需要
        outputs = net(inputs.squeeze(1))  # RNN
        # outputs = net(inputs)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.6f} %')
