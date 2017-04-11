#coding:utf-8
#ニューラルネットで予測するモデル

from neuralnet import neuralnetC
import pandas as pd

class predict_neuralnet(neuralnetC):
    def read_data(self):
        self.train = []
        self.test=[]
        self.train.append(pd.read_csv('data/point_train/tyuou_' + str(self.maxdistance) + '_train.csv'))
        self.train.append(pd.read_csv('data/point_train/keihintohoku_' + str(self.maxdistance) + '_train.csv'))
        self.train.append(pd.read_csv('data/point_train/keiyou_' + str(self.maxdistance) + '_train.csv'))
        self.train.append(pd.read_csv('data/point_train/uchibou_' + str(self.maxdistance) + '_train.csv'))
        self.train.append(pd.read_csv('data/point_train/saikyoukawagoe_' + str(self.maxdistance) + '_train.csv'))
        self.test.append(pd.read_csv('data/point_train/sotobou_' + str(self.maxdistance) + '_test.csv'))
        self.test.append(pd.read_csv('data/point_train/utsunomiya_' + str(self.maxdistance) + '_test.csv'))
        self.test.append(pd.read_csv('data/point_train/yamanote_' + str(self.maxdistance) + '_test.csv'))
        self.test.append(pd.read_csv('data/point_train/syonan_' + str(self.maxdistance) + '_test.csv'))
        self.test.append(pd.read_csv('data/point_train/takasaki_' + str(self.maxdistance) + '_test.csv'))
        self.alldata=pd.concat([self.train[0],self.train[1],self.train[2],self.train[3],self.train[4],self.test[0],self.test[1],self.test[2],self.test[3],self.test[4]])


if __name__=='__main__':
    my=predict_neuralnet()
    my.read_data()
    print(my.alldata)