import numpy as np
import os
def get_min_per(matrix,percent):
    m,n = np.shape(matrix)
    l=[]
    for i in range(m):
        for j in range(n):
            l.append(matrix[i][j])
    l = sorted(l)
    per=int(len(l)*percent)
    return l[per]

def NMF(matrix,percent):
    m,n = np.shape(matrix)
    per=get_min_per(matrix,percent)
    for i in range(m):
        for j in range(n):
            if matrix[i][j]<=per:
                matrix[i][j]=0
    return matrix

def Deal_data(infile,outfile,percent,type):
    a = np.loadtxt(infile ,dtype=type, delimiter=',')  # 以float加载txt为矩阵形式
    b=NMF(a,percent)
    file=open(outfile,'w')
    out_str=''
    for line in b:
        line_str=''
        for each in line:
            line_str = line_str + '{},'.format(each)
        out_str = out_str + line_str[:-1]+'\n'
    file.write(out_str)


FilePath=[None]*3
FilePath[0] = os.path.abspath('.') + "\Haberman Data\\"  # data.txt是原数据
FilePath[1] = os.path.abspath('.') + "\Iris Data\\"  # data.txt是原数据
FilePath[2] = os.path.abspath('.') + "\Ionosphere Data\\"  # data.txt是原数据

Deal_data(FilePath[0]+'data.txt',FilePath[0]+'nmf_data.txt',0.1,int)
Deal_data(FilePath[1]+'data.txt',FilePath[1]+'nmf_data.txt',0.1,float)
Deal_data(FilePath[2]+'data.txt',FilePath[2]+'nmf_data.txt',0.1,float)




