import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建数据
x = np.linspace(-5, 5, 100)  # x范围
y = np.linspace(-5, 5, 100)  # y范围
x, y = np.meshgrid(x, y)  # 创建网格

# 高度函数，使用二维高斯函数
z = np.exp(-0.1*(x**2 + y**2))

# 创建3D图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis')

plt.show()
