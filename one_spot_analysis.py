#coding:utf-8
#各路線での観測地点との距離の相関検証

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from itertools import compress
import numpy as np
from operator import itemgetter
import pprint
from calc_distance import pt_distance
import seaborn as sns
import matplotlib.pyplot as plt

class RandomForestC():
    def __init__(self):
        self.data=pd.read_csv('data/connected_train.csv')
        self.min = 1e-15
        self.trees=10
        self.features='auto'
        self.trainname='tyuou'
    def model_create(self):
        self.model=RandomForestClassifier(n_jobs=-1,verbose=1,n_estimators=self.trees,max_features=self.features,random_state=0)
    def cross_validation(self,K=10):
        split_length=int(1.0*len(self.data)/K)
        splited_data=[]
        for i in range(K):
            splited_data.append(self.data.ix[i*split_length:(i+1)*split_length])

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
            test=splited_data[i]

            #モデル初期化
            self.model_create()
            #訓練
            self.fit(train.ix[:,6:],train[self.trainname])
            #予測
            pred=self.predict(test.ix[:,6:])
            logloss+=self.logloss(pred,test[self.trainname].as_matrix())
            print('途中スコア：'+str(logloss))
            score+=logloss
        score/=K
        print('最終スコア：'+str(score))
    def df_merge(self,data):
        tmp=pd.DataFrame()
        for i in data:
            tmp=tmp.append(i)
        return tmp
    def fit(self,x_train,y_train):
        self.model.fit(x_train,y_train)
    def predict(self,x_test):
        pred=self.model.predict_proba(x_test)
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
    def set_trees(self,num):
        self.trees=num
    def set_features(self,method):
        self.features=method
    def set_trainname(self,name):
        self.trainname=name
    #特徴の重要度算出
    def get_feature_importance(self,limit=10):
        imp=self.model.feature_importances_
        names=self.data.ix[:,6:]
        names=names.columns
        tmp = list(zip(names, imp))
        tmp.sort(key=itemgetter(1), reverse=True)
        one=dict(tmp)
        one=sorted(one.items(), key=lambda x: x[1], reverse=True)
        final=[]
        for i in one:
            final.append(list(i))
        for i in range(len(final)):
            final[i].append(pt_distance('tyuou',int(final[i][0].split('_')[0])))

        temp=[]
        prec=[]
        wind=[]
        mwind=[]

        for i in range(len(final)):
            if final[i][0].split('_')[1] == 'temp':
                temp.append(final[i][1:])
            elif final[i][0].split('_')[1] == 'prec':
                prec.append(final[i][1:])
            elif final[i][0].split('_')[1] == 'wind':
                wind.append(final[i][1:])
            elif final[i][0].split('_')[1] == 'mwind':
                mwind.append(final[i][1:])
        temp = np.array(temp).T
        prec = np.array(prec).T
        wind = np.array(wind).T
        mwind = np.array(mwind).T

        sns.set(font='TakaoExMincho')
        sns.jointplot(temp[1],temp[0],kind='reg')
        plt.title(self.trainname+'の温度と地点距離の関係')
        plt.show()
        sns.jointplot(prec[1], prec[0],kind='reg')
        plt.title(self.trainname + 'の降水量と地点距離の関係')
        plt.show()
        sns.jointplot(wind[1], wind[0],kind='reg')
        plt.title(self.trainname + 'の風速と地点距離の関係')
        plt.show()
        sns.jointplot(mwind[1], mwind[0],kind='reg')
        plt.title(self.trainname + 'の最大瞬間風速と地点距離の関係')
        plt.show()

        with open('result.txt', 'w') as fout:
            fout.write(pprint.pformat(final))



if __name__=='__main__':
    myclass=RandomForestC()
    myclass.set_trainname('saikyoukawagoe')
    myclass.set_trees(600)
    myclass.set_features('auto')
    myclass.cross_validation(K=10)
    myclass.get_feature_importance()



