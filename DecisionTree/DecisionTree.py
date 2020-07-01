"""
Time : 2020-7-01 11:37
Author : lighteningzhang
Software : jupyter notebook
Email : lighteningzhang@sjtu.edu.cn
Problem: Recursively create tree
Accuracy: 81.3%
Notes: 用np.log2和log结果不同
"""
import pandas as pd
import numpy as np
def load_data():
    data_path = r'../Mnist/'
    train_data = pd.read_csv(data_path+'mnist_train.csv', header = None)
    test_data = pd.read_csv(data_path+'mnist_test.csv ', header = None)
    xtrain = train_data.loc[:, 1:]
    ytrain = train_data.loc[:, 0]
    xtest = test_data.loc[:, 1:]
    ytest = test_data.loc[:, 0]
    xtrain[xtrain<128] = 0
    xtrain[xtrain>=128] = 1
    xtest[xtest<128] = 0
    xtest[xtest>=128] = 1
    xtrain, xtest  = xtrain.to_numpy(), xtest.to_numpy()
    
    return xtrain, ytrain, xtest, ytest

def majorClass(labels):
    '''
    计算所有类中占比最大的类
    '''
    dic = {}
    for l in labels:
        if l in dic:
            dic[l] += 1
        else:
            dic[l] = 1
#  dic:
#  key: label, val: cnt
    res_dic = sorted(list(dic.items()), key = lambda it: it[1], reverse = True)
    return res_dic[0][0]
def entropy(prob_list):
    return sum([-prob*np.log2(prob) for prob in prob_list])
def cal_H(ytrain):
    '''
    计算当前特征的熵增
    '''
    tol, dic = 0, {}
    for y in ytrain:
        if y in dic:
            dic[y] += 1
        else:
            dic[y] = 1
        tol += 1
    prob_ls = []
    for k in dic.keys():
        prob_ls.append(dic[k]/tol)
    return entropy(prob_ls)

def cal_D_A(xfeature, ytrain):
    featureSet = set(xfeature)
    D_A = 0
    for fea in featureSet:
        D_A = D_A + xfeature[fea == xfeature].size/xfeature.size*cal_H(ytrain[xfeature==fea])
    return D_A

def sel_fea(xfeas, ytrain):
    n, m = xfeas.shape
    HD = cal_H(ytrain)
#     print("m: ", m)
    max_delt, max_fea = 0, 0
    for i in range(m):
        fea = xfeas[:, i]
        D_A = cal_D_A(fea, ytrain)
        if HD-D_A>max_delt:
            max_delt = HD-D_A
            max_fea = i
    return max_fea, max_delt


def getSubData(xtrain, ytrain, fea, a):
    '''
    获取分割后的特征和标签
    :param xtrain: 训练数据集
    :param ytrain: 标签
    :param fea: 哪维特征
    :param a: 标签值是多少
    '''
    ret_labels, ret_feas = [], [] 
    for i in range(len(xtrain)):
        if xtrain[i][fea] == a:
#  方便合并
            ret_feas.append(list(xtrain[i][:fea])+list(xtrain[i][fea+1:]))
            ret_labels.append(ytrain[i])
    return np.array(ret_feas), np.array(ret_labels)

def createTree(xtrain,ytrain):
    '''
    建树
    :param xtrain: 训练数据
    :param ytrain: 标签
    :return: 新的子节点的值
    '''
    epsilon = 0.2
 
    print("Begin creating tree...... ")             
    clas = set([i for i in ytrain])
    if clas == 1:
        return ytrain[0]
    if len(xtrain) == 0:
        return majorClass(ytrain)
    max_fea, max_delt = sel_fea(xtrain, ytrain)
    if max_delt<0.2:
        return majorClass(ytrain)
#     print("xtrain type: ", type(xtrain), " max feature: ", max_fea)
    types = set([i for i in xtrain[:, max_fea]])
    tree_dic = {max_fea: {}}
    print("types: ", types)
    for t in types:
        rest_xtrain, rest_ytrain= getSubData(xtrain, ytrain, max_fea, t)
        tree_dic[max_fea][t] = createTree(rest_xtrain, rest_ytrain)
    return tree_dic

def predict(xtest, tree):
    '''
    预测xtest的标签
    :param xtest: 测试样本
    :param tree: 决策树
    :return: 预测结果
    '''
    while 1:
        (key, value),  = tree.items()
        if type(tree[key]) == dict:
            cur_data = xtest[key]
# 每次使用了特征还要将其删除
            del xtest[key]
            tree = value[cur_data]
            if type(tree).__name__ == 'int64':
                return tree
        else:
            return value


def model_test(xtest, ytest, tree):
    '''
    测试
    '''
    err_cnt = 0
    for i in range(len(xtest)):
        if ytest[i] != predict(list(xtest[i]), tree):
            err_cnt += 1
            print("Current accuracy: %.3f"%(1-err_cnt/(i+1)))
    return 1-err_cnt/len(xtest)

import time
if __name__ == '__main__':
    start = time.time()
    xtrain, ytrain, xtest, ytest = load_data()
    tree = createTree(xtrain, ytrain)
    print("The tree is: ", tree)
    acc = model_test(xtest, ytest, tree)
    print("Accuracy score: ", acc)
    end = time.time()