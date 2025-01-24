import numpy as np
import math as mt
import matplotlib.pyplot as plt
import seaborn as sns

#初始序列
X0 = np.array([44,37,36,36,48,46,39,40])

#累加序列
X1 = X0.cumsum()

#权值计算
rho = [X0[i]/X1[i-1] for i in range(1,len(X0))] 
rho_ratio = [rho[i+1]/rho[i] for i in range(len(rho)-1)]

print("rho:",rho)
print("rho_ratio",rho_ratio)

flag = True


########################以下两种检验二选一即可######################
#光滑比检验
for i in range(1,len(rho)-1):
    if rho[i] > 0.5 or rho[i+1] / rho[i] >= 1:
        flag = False
if rho[-1] > 0.5:
    flag = False

if flag:
    print("Check one passed")
else:
    print("Check one failed")

#级比检验
for i in range(len(X0)-1):
    l = X0[i]/X0[i+1]
    if l <= mt.exp(-2/(len(X0)+1)) or l >= mt.exp(2/(len(X0)+1)):
        break;
if i ==  len(X0)-2 and l > mt.exp(-2/len(X0)+1) and l < mt.exp(2/(len(X0)+1)):
    print("Check two passed")
else:
    print("Check two failed")
    #不通过处理
    j = 1
    while True:
        YO = [k+j for k in X0]
        j += 1
        for m in range(len(YO) - 1):
            l = YO[m] / YO[m+1]
            if l > mt.exp(-2 / (len(X0)+1)) and l < mt.exp(2 / (len(X0)+1)):
                b = True
            else:
                b = False
                break
        if b == True:
            print("Get a new array")
            c = j-1
            print("Add value is:",c)
            break
        else:
            continue

###########################模型求解################################
X1 = X0.cumsum()
print("X0:",X0)
print("X1:",X1)

#紧邻均值序列
z = []
j = 1
while j < len(X1):
    num = (X1[j]+X1[j-1])/2
    z.append(num)
    j = j+1

#最小二乘法
Y = []
x_i = 0
while x_i < len(X0)-1:
    x_i += 1
    Y.append(X0[x_i])
Y = np.mat(Y)
print("Y:",Y)
Y = Y.reshape(-1,1)

B = []
b = 0
while b < len(z):
    B.append(-z[b])
    b += 1
B = np.mat(B)
B = B.reshape(-1,1)
c = np.ones((len(B),1))
B = np.hstack((B,c))
print("B:",B)

#求解参数
alpha = np.linalg.inv(B.T.dot(B)).dot(B.T).dot(Y)
a = alpha[0,0]
b = alpha[1,0]
print("alpha=",alpha)
print("a=",a)
print("b=",b)

#预测模型
GM = []
GM.append(X0[0])
did = b/a
k=1
while k < len(X0):
    GM.append((X0[0]-did)*mt.exp(-a*k)+did)
    k += 1

#做差得到预测序列
G = []
# G.append(X0[0])
g = 1
while g < len(X0):
    G.append(round(GM[g]-GM[g-1]))
    g += 1

print("predict arrays:",G)

######################绘图##########################

start_year = 2008

years = [i*4+start_year for i in range(1,len(X0)+1)]

future = [i*4+start_year for i in range(len(X0)+1,2*len(X0))]

all_years = years+future
all_gold = np.concatenate((X0, G))  # 合并金牌数

# 设置 Seaborn 样式
sns.set(style="whitegrid")  # 设置背景为白色网格

# 创建一个图形
plt.figure(figsize=(10, 6))

# 使用 Seaborn 的 lineplot 绘制历史数据
sns.lineplot(x=all_years, y=all_gold, color='red', label='Gold medals all_years', linewidth=2)

# 使用 Seaborn 的 lineplot 绘制预测数据
sns.lineplot(x=future, y=G, color='blue', label='Future gold medals', linewidth=2)

# 图表设置
plt.xlabel('Olympics Year', fontsize=12)
plt.ylabel('Gold Medals', fontsize=12)
plt.title('The Gold Medals Prediction of United States using GM(1,1)', fontsize=14)

# 图例
plt.legend()

# 展示图形
plt.show()

