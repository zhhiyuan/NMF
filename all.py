import numpy as np
import sklearn.model_selection as train
from sklearn import tree
from sklearn.metrics import accuracy_score
import os
#处理旧数据
from distrbut import Dis
from  NMF import nmf
from Compare import compare
if __name__ == '__main__':
    n = [34, 30, 3, 5]
    FilePath = [None] * 4
    FilePath[0] = os.path.abspath('.') + "\\Ionosphere Data"
    FilePath[1] = os.path.abspath('.') + "\\WBC"
    FilePath[2] = os.path.abspath('.') + "\\Haberman Data"
    FilePath[3] = os.path.abspath('.') + "\\Iris Data"
    u = 0.02
    for i in range(4):
        print(FilePath[i])
        filename = open(FilePath[i] + '\\data.txt')
        data = np.loadtxt(filename, dtype=float, delimiter=',')
        x, y = np.split(data, indices_or_sections=(1,), axis=1)
        # 后十个为属性值，第一个为标签
        x, y = y[:, 1:], x
        # 抽取0.6作为训练集
        x_train, x_test, y_train, y_test = train.train_test_split(x, y, random_state=1, train_size=0.6)

        clf = tree.DecisionTreeClassifier(max_depth=20)
        clf.fit(x_train, y_train.ravel())
        o_y_hat = clf.predict(x_test)
        # 原始数据测试集的准确率
        score = accuracy_score(o_y_hat, y_test)
        # 不同e情况下
        for j in range(1, n[i]):
            rate = j / n[i]
            if rate>0.6:
                continue
            print('e={}情况下的detail R：'.format(round(rate,2)))
            RNN_x_test = Dis(x_test, j, n[i], rate)
            RNN_y_hat = clf.predict(RNN_x_test)
            RNN_score = accuracy_score(RNN_y_hat, y_test)
            RNN_detail_R = RNN_score - (1 - u) * score
            print('RNN_detail_R:{}'.format(RNN_detail_R))
            print('RNN 五个值：')
            compare(x_test, RNN_x_test)
            NMF_x_test = nmf(x_test, rate)
            NMF_y_hat = clf.predict(NMF_x_test)
            NMF_score = accuracy_score(NMF_y_hat, y_test)
            NMF_detail_R = NMF_score - (1 - u) * score
            print('NMF_detail_R:{}'.format(NMF_detail_R))
            compare(x_test, RNN_x_test)
            print('NMF 五个值：')
            compare(x_test, NMF_x_test)

            print()
        print()






