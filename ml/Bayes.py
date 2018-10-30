from sklearn.naive_bayes import GaussianNB  #高斯朴素贝叶斯
import numpy as np
import sklearn.model_selection as train
from sklearn.metrics import accuracy_score
import os

FilePath = [None]*2
FilePath[0] = os.path.abspath('..') + "\\Ionosphere Data"
FilePath[1] = os.path.abspath('..') + "\\final_data"

def loadData(filename,type):
    data = np.loadtxt(filename, dtype=type, delimiter=',',skiprows=2)
    x,y=np.split(data,indices_or_sections=(1,),axis=1)
    #后十个为属性值，第一个为标签
    x ,y= y[:,1:],x
    #前十个为属性值
    x_train,x_test,y_train,y_test=train.train_test_split(x,y,random_state=1,train_size=0.6)
    #随机划分训练集与测试集
    return x_train,x_test,y_train,y_test

def Train(x_train,y_train):
    clf = GaussianNB()
    clf.fit(x_train, y_train.ravel())
    return clf


def Test(x_train,x_test,y_train,y_test,clf):
    if clf is None:
        raise IOError("Must input a clf!")
    y_hat = clf.predict(x_train)
    score = accuracy_score(y_hat, y_train)
    print('训练集准确率：{}'.format(score))
    y_hat=clf.predict(x_test)
    score=accuracy_score(y_hat,y_test)
    print('测试集准确率：{}'.format(score))


#返回文件夹数组
def getFilepath(FilePath):
    parents = os.listdir(FilePath)
    files = []
    #获取data文件夹下除了data.txt的其他文件的路径(加噪后的数据文件)

    for parent in parents:
        if "50" in parent:
            files.append(parent)
        if "30" in  parent:
            files.append(parent)
    files.append("nmf_data.txt")
    files.append('data.txt')
    return files

if __name__ == '__main__':
    n=4
    for Path in FilePath:
        print(Path)
        datas = getFilepath(Path)
        for i in range(len(datas)):
            x_train1, x_test1, y_train1, y_test1 = loadData(Path+ '\\'+ datas[i], float)
            clf1 = Train(x_train1, y_train1)
            print(datas[i]+'数据')
            Test(x_train1, x_test1, y_train1, y_test1, clf1)

        print()
        print()
