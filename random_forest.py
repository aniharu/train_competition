#coding:utf-8
#ランダムフォレストによる特徴選択

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from itertools import compress
import numpy as np
from operator import itemgetter
import pprint

class RandomForestC():
    def __init__(self):
        self.data=pd.read_csv('data/noprec_connected_train.csv')
        self.min = 1e-15
        self.trees=10
        self.features='auto'
        self.minimum=0
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
            self.fit(train.ix[:,6:],train[['tyuou','keihintohoku','keiyou','uchibou','saikyoukawagoe']])
            #予測
            pred=self.predict(test.ix[:,6:])
            for i in ('tyuou','keihintohoku','keiyou','uchibou','saikyoukawagoe'):
                logloss+=self.logloss(pred,test[i].as_matrix())
            logloss/=5
            print('途中スコア：'+str(logloss))
            score+=logloss
        score/=K
        print('最終スコア：'+str(score))
        return score
    def df_merge(self,data):
        tmp=pd.DataFrame()
        for i in data:
            tmp=tmp.append(i)
        return tmp
    def fit(self,x_train,y_train):
        tname=('tyuou','keihintohoku','keiyou','uchibou','saikyoukawagoe')
        for i in range(5):
            self.model.fit(x_train,y_train[tname[i]])
    def predict(self,x_test):
        pred=self.model.predict_proba(x_test)
        return self.predict_modify(pred)
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
    def set_minimum(self,minimum):
        self.minimum=minimum
    #特徴の重要度算出
    def get_feature_importance(self,limit=10):
        imp=self.model.feature_importances_
        names=self.data.ix[:,6:]
        names=names.columns
        tmp = list(zip(names, imp))
        tmp.sort(key=itemgetter(1), reverse=True)
        one=dict(tmp)
        one=sorted(one.items(), key=lambda x: x[1])
        with open('result.txt', 'w') as fout:
            fout.write(pprint.pformat(one))
    #予測値の修正
    def predict_modify(self,data):
        for i in range(len(data)):
            for j in range(4):
                if data[i,j]==0:
                    data[i,j]+=self.minimum
                    data[i,np.argmax(data[i])]-=self.minimum

        print(np.min(data))

        return data



if __name__=='__main__':
    result=[]
    for i in [0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001,0.00000001]:
        myclass=RandomForestC()
        myclass.set_minimum(i)
        myclass.set_trees(600)
        myclass.set_features('log2')
        result.append(myclass.cross_validation(K=10))
    print(result)



