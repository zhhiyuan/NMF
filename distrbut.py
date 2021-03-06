# 梅森旋转算法   -xlxw
# 参考:mersenne twister from wikipedia

# import
from time import time
import numpy as np
import os
import random
import math
from numpy import nan as NaN
# var
index = 624
MT = [0] * index
# MT[0] ->seed
# 数据目录
FilePath = os.path.abspath('..') + "\\final_data"

def inter(t):
    return (0xFFFFFFFF & t)  # 取最后32位->t


def twister():
    global index
    for i in range(624):
        y = inter((MT[i] & 0x80000000) + (MT[(i + 1) % 624] & 0x7fffffff))
        MT[i] = MT[(i + 397) % 624] ^ y >> 1
        if y % 2 != 0:
            MT[i] = MT[i] ^ 0x9908b0df
    index= 0


def exnum():
    global index
    if index >= 624:
        twister()
    y = MT[index]
    y = y ^ y >> 11
    y = y ^ y << 7 & 2636928640
    y = y ^ y << 15 & 4022730752
    y = y ^ y >> 18
    index = index + 1
    return inter(y)


def mainset(seed):
    MT[0] = seed  # seed
    for i in range(1, 624):
        MT[i] = inter(1812433253 * (MT[i - 1] ^ MT[i - 1] >> 30) + i)
    return exnum()


# BOX-MULLER 伪随机数生成方法
# 处理[0,1]范围的数据
def boxmuller():
    # boxmuller()[0]
    while (1):
        summa = 1
        size = 1
        x = np.random.uniform(size=size)
        y = np.random.uniform(size=size)
        z = np.sqrt(-2 * np.log(x)) * np.cos(2 * np.pi * y)
        q = z * summa
        if q > 0:
            return q

#根据属性个数生成随机数
def Random_list(num,n):
    random_list = []
    while len(random_list) < num:   #随机数个数
        y = random.uniform(0,n-1)  # 0,n之间抽样随机数
        if round(y) not in random_list:
            random_list.append(round(y))
    return random_list    #返回随机数组


def Dis(old_data,num,n):
    '''
    我们的干扰方法
    :param old_data: 数据列表
    :param num: 干扰数目
    :param n: 属性个数
    :return:
    '''
    abs_data=np.abs(old_data)   #用于干扰公式
    data=np.matrix.tolist(old_data)
    for j in range(len(data)):  #对于每一行
        list_data = Random_list(num,n)   #随机数组
        for i in list_data:    #对随机位置的数干扰处理
            idata = float(data[j][i])    #取出该位置的数
            #根据给定比列确定随机数生成区间，[min_data,max_data]

            #带根号
            sum=32*(np.mean(abs_data[:,i]) )
            rate = math.sqrt((len(old_data)*abs_data[j][i]) / sum)   #求随机列i的平均值

            '''
            sum =  2*(np.mean(abs_data[:, i]))
            rate = (len(old_data) * abs_data[j][i]) / sum # 求随机列i的平均值
            '''
            min_idata = idata * (1-rate)
            max_idata = idata * (1+rate)
            if min_idata>max_idata:
                max_idata,min_idata = min_idata,max_idata
            so = mainset(int(time())) / (2 ** 32 - 1)
            #O线性同余法计算随机数
            #在min_data，max_data之间取一个随机数替代之前的数
            new_idata = min_idata + float((max_idata - min_idata) * so)
            data[j][i] = round(new_idata,3)   #保留三位小数
            if math.isnan(data[j][i]):
                data[j][i]=0
    return data


def dis_add(old_data, num ,n):
    '''
    加性干扰
    :param old_data: 数据列表
    :param num: 干扰数目
    :param n: 属性个数
    :return:
    '''
    dis_range =   50    #加性的范围
    data = np.matrix.tolist(old_data)
    for j in range(len(data)):  # 对于每一行
        list_data = Random_list(num, n)  # 随机数组
        for i in list_data:  # 对随机位置的数干扰处理
            idata = float(data[j][i])  # 取出该位置的数
            # 根据给定比列确定随机数生成区间，[min_data,max_data
            min_idata = idata + dis_range
            max_idata = idata - dis_range
            if min_idata > max_idata:
                max_idata, min_idata = min_idata, max_idata
            so = mainset(int(time())) / (2 ** 32 - 1)
            # O线性同余法计算随机数
            # 在min_data，max_data之间取一个随机数替代之前的数
            new_idata = min_idata + float((max_idata - min_idata) * so)
            data[j][i] = round(new_idata, 3)  # 保留三位小数
            if math.isnan(data[j][i]):
                data[j][i] = 0
    return data


def dis_mul(old_data, num ,n):
    '''
       加性干扰
       :param old_data: 数据列表
       :param num: 干扰数目
       :param n: 属性个数
       :return:
       '''
    dis_range = 0.8  # 乘性的范围

    data = np.matrix.tolist(old_data)
    for j in range(len(data)):  # 对于每一行
        list_data = Random_list(num, n)  # 随机数组
        for i in list_data:  # 对随机位置的数干扰处理
            idata = float(data[j][i])  # 取出该位置的数
            # 根据给定比列确定随机数生成区间，[min_data,max_data
            new_idata = idata * (1+dis_range)
            data[j][i] = round(new_idata, 3)  # 保留三位小数
            if math.isnan(data[j][i]):
                data[j][i] = 0
    return data


if __name__ == '__main__':
    # 第一个参数是随机属性的个数，第二个参数是比率
    # 比率k：在n(1-k)和n(1+k)之间取一个随机数
    data=[[1,2,3],[1,2,5],[100,2,1]]
    print(data)
    print(Dis(np.array(data),3,3))
    '''
    n = 30
    for i in range(1,n):
        add_main(i,n,50,'add {},30.txt'.format(i))
        print('one done!')

'''