# %% md

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

"""************************************代价函数*********************************"""

def costFunction(X, y, theta):
    inner = np.power(X @ theta - y, 2)
    return np.sum(inner) / (2 * len(X))

"""************************************梯度下降*********************************"""
def gradientDescent(X, y, theta, alpha, iters):
    costs = []

    for i in range(iters):
        theta = theta - (X.T @ (X @ theta - y)) * alpha / len(X)
        cost = costFunction(X, y, theta)
        costs.append(cost)

        if i % 100 == 0:
            print(cost)

    return theta, costs

"""************************************主程序入口******************************"""

'''
读取数据
'''
data = pd.read_csv('temperture.txt', names=['distance', 'environment','test','true']) #读取数据并创建data对象,给所有列向量贴上标签

# # 调试用——输出前五行数据
#data.head()
# # 调试用—— 输出后五行数据
# data.tail()
# # 调试用——统计数据
# data.describe()
'''
输出数据
'''
# # 输出统计数据类型及内存
# data.info()
# #输出二维数据集——点状图
# data.plot.scatter('distance', 'environment','test','true', label='test')
# plt.show()
'''
数据处理
'''
# #在数据集第一列加入一列全为1的列向量组成特征集合
data.insert(0, 'ones', 1)
data.head()
#去掉结果，组成特征集
X = data.iloc[:, 0:-1]
X.head()

#取数据集最后一列为结果向量
y = data.iloc[:, -1]
y.head()

# #以array形式返回指定column的所有取值   （显示所有特征组）
X = X.values

# #调试用——显示矩阵列数行数
# t= X.shape

# #以array形式返回指定column的所有取值   （显示所有结果组）
y = y.values
# t=y.shape
#
#行向量转列向量
y = y.reshape(len(y), 1)
# t=y.shape

'''
计算代价函数
'''

# #theta初始化
theta = np.zeros((4, 1))
#调试用——显示矩阵列数行数
T=theta.shape

#计算代价函数
cost_init = costFunction(X, y, theta)
#输出初始值
#print(cost_init)



'''
梯度下降算法
'''
# 学习率alpha
alpha = 0.00001
#迭代次数 iters
iters = 8000
#进行梯度下降
theta, costs = gradientDescent(X, y, theta, alpha, iters)
'''
显示梯度下降
'''
# #fig为返回的图像，ax为返回的坐标系（为一个数组，如果是多个坐标系的话）
# fig, ax = plt.subplots()
# #画出坐标系，必须使用arange不能使用range，range是一个list，arrange是一个数据
# ax.plot(np.arange(iters), costs)
# #设置坐标系名字和图像名字
# ax.set(xlabel='iters',
#        ylabel='cost',
#        title='cost vs iters')
# #画出图像
# plt.show()

print("最终theta结果为\r\n",theta)
dis=input("请输入距离：")
env=input("请输入环境温度")
tes=input("请输入测试温度")
truetem = [1,float(dis),float(env),float(tes)] @ theta
print("预测的体温为",truetem)
input("Please Enter to exit")


# '''
# 画出数据集
# '''
# x = np.linspace(y.min(), y.max(), 100)
# y_ = theta[0, 0] + theta[1, 0] * x
#
# fig, ax = plt.subplots()
# #画点
# ax.scatter(X[:, 1], y, label='training data')
# #画线
# ax.plot(x, y_, 'r', label='predict')
# #显示数据标签
# ax.legend()
# #设置坐标系名字
# ax.set(xlabel='populaiton',
#        ylabel='profit')
# #画出图像
# plt.show()
#

