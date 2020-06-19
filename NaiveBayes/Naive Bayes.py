"""
Time : 2020-6-19 23:56
Author : lighteningzhang
Software : jupyter notebook
Email : lighteningzhang@sjtu.edu.cn
Problem: Represent the P(Xij = m/Yi = ck)
"""
import pandas as pd
import numpy as np
import time
data_path = r'../Mnist/'
def load_data():
    dtrain = pd.read_csv(data_path+'mnist_train.csv', header = None)
    dtest = pd.read_csv(data_path+'mnist_test.csv', header = None)
    xtrain = dtrain.loc[:,1:]
    ytrain = dtrain.loc[:, 0]
    xtest = dtest.loc[:, 1:]
    ytest = dtest.loc[:, 0]
#  图像二值化，因为条件概率下y要对每个feature的值进行预测
#  max p(y=ck)*p(x1,x2,x3,...,xn/y=ck)
    xtrain[xtrain<128] = 0
    xtrain[xtrain>=128] = 1
    xtest[xtest<128] = 0
    xtest[xtest>=128] = 1
    return xtrain, ytrain, xtest, ytest


# In[170]:


def calProb(xtrain, ytrain, lamb = 1):
    xtrain = xtrain.to_numpy()
    ytrain = ytrain.to_numpy()
    n, m = xtrain.shape
    Ixy = np.zeros((m,2, 10))
    
    Iy = np.zeros(10)
    Py = np.zeros(10)
    Px_y = np.zeros([m, 2, 10])
    for k in range(10):
        Iy[k] = np.sum(ytrain==k)
        Py[k] = np.log((Iy[k]+lamb)/(n+10*lamb))
    for i in range(n):
#         print(i)
        for j in range(m):
#  种类为 ytrain[i], 特征所在的坐标是j， 特征坐标对应的值是 xtrain[i][j]
            Ixy[j][xtrain[i][j]][ytrain[i]] += 1
    for i in range(m):
        for k in range(10):
            Px_y[i][0][k] = np.log((Ixy[i][0][k]+lamb)/(Iy[k]+2*lamb))
            Px_y[i][1][k] = np.log((Ixy[i][1][k]+lamb)/(Iy[k]+2*lamb))
    return Py, Px_y


# In[190]:


def NaiveBayes(Py, Px_y, x):
    #x 为 要估计的样本
    _, d = x.shape
    x = np.array(x)
    classNum = 10
    Pclass = [0]*10
    for i in range(classNum):
        for j in range(d):
#            label yi条件下， 算x每个特征出现值 xj的概率
            Pclass[i] += Px_y[j][x[0][j]][i]+Px_y[j][x[0][j]][i]
            
        Pclass[i]+=Py[i]
#     np.argmax(Pclass)
    return Pclass.index(max(Pclass))


# In[191]:


def model_test(xtest, ytest, Py, Px_y):
    n, m = xtest.shape
    acc = 0
    xtest = np.mat(xtest)
    ytest = np.array(ytest)
    
    for i in range(n):
        cla = NaiveBayes(Py, Px_y, xtest[i])
        if cla == ytest[i]:
            acc += 1
        print("Current accuracy: %.2f"%(acc/(i+1)))
    return (acc/n)


# In[188]:


if __name__ == "__main__":
    xtrain, ytrain, xtest, ytest = load_data()
    start = time.time()
    Py, Px_y = calProb(xtrain, ytrain)
    acc = model_test(xtest, ytest, Py, Px_y)
    end = time.time()
    print("Time: %d | Final accuracy: %.2f"%(end-start,acc))

