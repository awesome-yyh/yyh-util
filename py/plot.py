import matplotlib.pyplot as plt


eve_type_spell = False

x_data = []
y_data = []
plt.plot(x_data, y_data, color='lightcoral', label='xxx')  # lightcoral, lightblue, lightgreen, darkred, darkblue, darkgreen
plt.scatter(x_data, y_data, color='lightcoral')  # 散点图

x_data = []
y_data = []
plt.plot(x_data, y_data, color='lightblue', label='xx')
plt.scatter(x_data, y_data, color='lightblue')  # 散点图

# plt.plot(x_data, np.array(Weights * x_data + biases), label='Fitted line')
plt.legend()  # 显示图示

plt.xlabel("step")
plt.ylabel("F1")
plt.title("my plot")
# plt.ylim(-1.2,1.2)  # 设置Y轴的范围

plt.show()
