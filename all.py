import numpy as np
import sklearn.model_selection as train
from sklearn import tree
from sklearn.metrics import accuracy_score
import os
import csv
#处理旧数据
from distrbut import Dis
from  NMF import nmf
from Compare import compare
if __name__ == '__main__':
    n = [34]
    FilePath = [None] * 1
    path = os.path.abspath('.') + "\\"
    #FilePath[0] = "WBC"     #n=30
    #FilePath[0] = "Iris Data"      #n=3
    FilePath[0] = "Ionosphere Data"    #n=34
    for i in range(1):
        print(FilePath[i])
        filename = open(path + FilePath[i] + '\\data.txt')
        data = np.loadtxt(filename, dtype=float, delimiter=',')
        x, y = np.split(data, indices_or_sections=(1,), axis=1)
        # 后十个为属性值，第一个为标签

        if FilePath[i] == "Iris Data":
            x, y = y[:, 1:], x
        else:
            x, y = y[:, :], x
        # 抽取0.6作为训练集
        x_train, x_test, y_train, y_test = train.train_test_split(x, y, random_state=1, train_size=0.4)

        clf = tree.DecisionTreeClassifier(max_depth=20)
        clf.fit(x_train, y_train.ravel())
        o_y_hat = clf.predict(x_test)
        # 原始数据测试集的准确率
        score = accuracy_score(o_y_hat, y_test)
        # 不同e情况下
        with open(FilePath[i] + '.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for j in range(1, n[i]):
                write = []
                rate = j / n[i]

                if rate > 0.7:
                    continue
                print('e={}情况下的detail R：'.format(round(rate,2)))
                write.append(round(rate,2)) ###


                RNN_x_test = Dis(x_test, j, n[i])
                a=np.max(RNN_x_test)
                RNN_y_hat = clf.predict(RNN_x_test)
                RNN_score = accuracy_score(RNN_y_hat, y_test)
                RNN_detail_R = RNN_score -score
                print('RNN_detail_R:{}'.format(RNN_detail_R))
                print('RNN 五个值：')
                five_rnn = compare(x_test, RNN_x_test)
                print(five_rnn)
                write.append(RNN_detail_R)
                write += five_rnn
                write.append(0)


                x_test1=np.array(x_test)
                NMF_x_test = nmf(x_test1, j , n[i])
                NMF_y_hat = clf.predict(NMF_x_test)
                NMF_score = accuracy_score(NMF_y_hat, y_test)
                NMF_detail_R = NMF_score -score
                print('NMF_detail_R:{}'.format(NMF_detail_R))
                print('NMF 五个值：')
                five_nmf = compare(x_test, NMF_x_test)
                print(five_nmf)
                write.append(NMF_detail_R)
                write += five_nmf

                writer.writerow(write)
                print()
        print()






