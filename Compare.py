import numpy as np
import os

# a为M矩阵，b为M帽矩阵(加噪后)
def cVD(a, b):
    c = a - b   #计算两个矩阵的差
    # np.linalg.norm(c)为计算矩阵的范数
    return (np.linalg.norm(c)/np.linalg.norm(a))

#计算RP和RK
def cRP(a, b):
    q = len(a)    #q个对象，相当于行
    p = len(a[0])    #p个属性，相当于列
    sumRK = 0
    #i为行，j为列

    #获取按列排序后的索引值
    orda = np.argsort(a, axis=0)
    ordb = np.argsort(b, axis=0)
    ord = np.abs(orda - ordb)

    for i in range(q):
        for j in range(p):
            if orda[i][j]==ordb[i][j]:
                sumRK += 1

    RP = np.sum(ord) / (p * q)   #计算RP值
    RK = sumRK / (p * q)   #计算RK值
    return RP, RK

#计算cp和CK值
def cCP(a, b):
    p = len(a[0])  # p个属性，相当于列
    sumCK = 0

    #每列索引值的均值
    orda = np.mean(np.argsort(a, axis=0), axis=0)
    ordb = np.mean(np.argsort(b, axis=0), axis=0)
    ord = np.abs(orda - ordb)

    for i in range(p):
        if orda[i]==ordb[i]:
            sumCK += 1

    CP = np.sum(ord) / p   #计算CP值
    CK = sumCK / p
    return CP, CK


#返回文件夹数组
def getFilepath(FilePath):
    parents = os.listdir(FilePath)
    files = []
    #获取data文件夹下除了data.txt的其他文件的路径(加噪后的数据文件)
    for parent in parents:
        if "50" in parent:
            files.append(parent)
    files.append("nmf_data.txt")
    return files


if __name__ == '__main__':
    FilePath = [None] * 3
    FilePath[0] = os.path.abspath('.') + "\Haberman Data\\"  # data.txt是原数据
    FilePath[1] = os.path.abspath('.') + "\Iris Data\\"  # data.txt是原数据
    FilePath[2] = os.path.abspath('.') + "\Ionosphere Data\\"  # data.txt是原数据
    for Path in FilePath:
        print(Path)
        datas = getFilepath(Path)  # 获取data文件夹下所有文件，即加噪后的数据
        a = np.loadtxt(Path + "data.txt", dtype=float, delimiter=',')  # 以float加载txt为矩阵形式
        for i in range(len(datas)):
            b = np.loadtxt(Path + datas[i], dtype=float, delimiter=',')
            # 分别计算五个参数
            VD = cVD(a, b)
            RP, RK = cRP(a, b)
            CP, CK = cCP(a, b)
            print("data.txt ", datas[i], " VD=", VD, ", RP=", RP, ", RK=", RK, ", CP=", CP, ", CK=", CK)
        print()
'''
#多个文件
datas = getFilepath()   #获取data文件夹下所有文件，即加噪后的数据
FilePath = os.path.abspath('.') + "\data" + "\data.txt"    #data.txt是原数据
a = np.loadtxt(FilePath, dtype=float, delimiter=',')    #以float加载txt为矩阵形式
for i in range(len(datas)):
    b = np.loadtxt(datas[i], dtype=float, delimiter=',')
    #分别计算五个参数
    VD = cVD(a, b)
    RP, RK = cRP(a, b)
    CP, CK = cCP(a, b)
    print("VD=", VD, ", RP=", RP, ", RK=", RK, ", CP=", CP, ", CK=", CK)
'''

'''
#单个文件
FilePatha = os.path.abspath('.') + "\data" + "\data.txt"  # data.txt是原数据
FilePathb = os.path.abspath('.') + "\data" + "\\1, 50.txt"  # data.txt是原数据

a = np.loadtxt(FilePatha, dtype=float, delimiter=',')  # 以float加载txt为矩阵形式
b = np.loadtxt(FilePathb, dtype=float, delimiter=',')
# 分别计算五个参数
VD = cVD(a, b)
RP, RK = cRP(a, b)
CP, CK = cCP(a, b)
print("VD=", VD, ", RP=", RP, ", RK=", RK, ", CP=", CP, ", CK=", CK)
'''

