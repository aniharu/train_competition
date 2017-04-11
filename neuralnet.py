#coding:utf-8
#ニューラルネットによる予測モデル

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from itertools import compress

class neuralnetC():
    def __init__(self):
        self.maxdistance=20
        self.min = 1e-15
    def read_data(self):
        self.train = []
        self.train.append(pd.read_csv('data/point_train/tyuou_' + str(self.maxdistance) + '_train.csv'))
        self.train.append(pd.read_csv('data/point_train/keihintohoku_' + str(self.maxdistance) + '_train.csv'))
        self.train.append(pd.read_csv('data/point_train/keiyou_' + str(self.maxdistance) + '_train.csv'))
        self.train.append(pd.read_csv('data/point_train/uchibou_' + str(self.maxdistance) + '_train.csv'))
        self.train.append(pd.read_csv('data/point_train/saikyoukawagoe_' + str(self.maxdistance) + '_train.csv'))
        self.alldata=pd.concat([self.train[0],self.train[1],self.train[2],self.train[3],self.train[4]])
    #zスコアを算出しかえす関数
    def zscore(self,data):
        df=pd.DataFrame(pd.get_dummies(data['state']).as_matrix(),columns=['none','people','machine','weather'])
        for i in ['temp','prec','wind','mwind']:
            df[i]=np.array((data[i] - self.alldata[i].mean()) / self.alldata[i].std())
        return df
    def df_merge(self, data):
        tmp = pd.DataFrame()
        for i in data:
            tmp = tmp.append(i)
        return tmp
    def cross_validation(self,K=10):
        split_length=int(1.0*len(self.train[0])/K)
        splited_data=[]

        for i in range(K):
            tmp = pd.DataFrame()
            for data in self.train:
                tmp=pd.concat([tmp,data.ix[i*split_length:(i+1)*split_length]],ignore_index=True)
            splited_data.append(tmp)

        score=0
        #実際にK-Fold開始
        for i in range(K):
            logloss=0
            print(str(i+1)+'\tof\t'+str(K))
            bfilter=[True] * K
            bfilter[i]=False
            #compressでboolによるフィルタ
            train=list(compress(splited_data,bfilter))
            train=self.df_merge(train)
            train=self.zscore(train)
            test=splited_data[i]
            act=test['state']
            test=self.zscore(test)

            #モデル初期化
            self.model_create()
            #訓練
            self.fit(train.ix[:,4:].as_matrix(),train.ix[:,:4].as_matrix())
            #予測
            pred=self.predict(test.ix[:,4:].as_matrix())
            logloss=self.logloss(pred,act.as_matrix())
            print('途中スコア：'+str(logloss))
            score+=logloss
        score/=K
        print('最終スコア：'+str(score))
        return score
    def model_create(self):
        self.model = Sequential()
        self.model.add(Dense(4, input_dim=4))
        self.model.add(Activation("relu"))
        self.model.add(Dense(4, input_dim=4))
        self.model.add(Activation("softmax"))
        self.model.compile(loss="categorical_crossentropy", optimizer="adam")
    def fit(self,x_train,y_train):
        self.model.fit(x_train,y_train,validation_split=0.1,verbose=0)
    def predict(self,x_test):
        #pred=self.model.predict_proba(x_test)
        pred = self.model.predict_proba(x_test,verbose=0)
        return pred
    def logloss(self,pred,act):
        logloss=0
        for i in range(len(act)):
            if pred[i,act[i]] < self.min:
                logloss+=np.log(self.min)
            else:
                logloss+=np.log(pred[i,act[i]])
        logloss/=len(act)
        logloss*=-1
        return logloss

if __name__=='__main__':
    my=neuralnetC()
    my.read_data()
    my.cross_validation()
