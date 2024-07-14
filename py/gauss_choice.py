import numpy as np
import matplotlib.pyplot as plt


# 定义一个高斯分布
mu = 0
sigma = 10

# 生成一个长度为 10 的序列
x = np.linspace(0, 50, 50)

# 计算每个元素的概率
p = 2 * np.exp(-(x - mu)**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)
print(sum(p))
plt.plot(x, p, color='lightcoral', label='xxx')

# 从序列中随机选择一个元素，并根据概率分布计算其概率
y = np.random.choice(x, p=p)

print(y)
plt.show()