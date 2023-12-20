import matplotlib.pyplot as plt
import math


eve_type_spell = False

x_data = [x for x in range(1, 10)]
y_data = [math.log(4*x**2) + 1 for x in x_data]
plt.plot(x_data, y_data, color='lightcoral', label='0')  # lightcoral, lightblue, lightgreen, darkred, darkblue, darkgreen
plt.scatter(x_data, y_data, color='lightcoral')  # 散点图

x_data = [x for x in range(1, 10)]
y_data = [math.log(4*x**2) for x in x_data]
plt.plot(x_data, y_data, color='lightgreen', label='1')  # lightcoral, lightblue, lightgreen, darkred, darkblue, darkgreen
plt.scatter(x_data, y_data, color='lightgreen')  # 散点图

x_data = [x for x in range(1, 10)]
y_data = x_data
plt.plot(x_data, y_data, color='lightblue', label='2')
plt.scatter(x_data, y_data, color='lightblue')  # 散点图

# plt.plot(x_data, np.array(Weights * x_data + biases), label='Fitted line')
plt.legend()  # 显示图示

plt.xlabel("step")
plt.ylabel("F1")
plt.title("my plot")
# plt.ylim(-1.2,1.2)  # 设置Y轴的范围

plt.show()
