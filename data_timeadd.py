#coding:utf-8
#時間の説明変数を追加するスクリプト

import pandas as pd
import pickle
import pprint
import numpy as np
import jholiday
from datetime import *

#休日と土日に1を返す関数
def doniti(date):
    if date.weekday()>=5:
        return 1
    else:
        if jholiday.holiday_name(year=date.year,month=date.month,day=date.day) != None:
            return 1
        else:
            return 0

class timeadd():
    def __init__(self,distance=20):
        with open('data/pickle/train_'+str(distance)+'km.pickle', 'rb') as f:
            self.train = pickle.load(f)
        with open('data/pickle/test_'+str(distance)+'km.pickle', 'rb') as f:
            self.test = pickle.load(f)
        self.timedf=pd.read_csv('data/train.csv',parse_dates=True,index_col=0)
        self.timedf = self.timedf.index.tolist() * 5
        #時間をcos,sinに変換
        self.cos = list(map(lambda x: np.cos(x.hour/24.0 * np.pi *2),self.timedf))
        self.sin = list(map(lambda x: np.sin(x.hour / 24.0 * np.pi * 2), self.timedf))

        self.doniti = list(map(lambda x: doniti(x),self.timedf))

        # # 曜日をcos,sinに変換
        # self.wcos = list(map(lambda x: np.cos(x.weekday() / 7.0 * np.pi * 2), self.timedf))
        # self.wsin = list(map(lambda x: np.sin(x.weekday() / 7.0 * np.pi * 2), self.timedf))
        # # 月をcos,sinに変換
        # self.mcos = list(map(lambda x: np.cos(x.month-1 / 12.0 * np.pi * 2), self.timedf))
        # self.msin = list(map(lambda x: np.sin(x.month-1 / 12.0 * np.pi * 2), self.timedf))

        self.train['hour_cos'] = self.cos
        self.train['hour_sin'] = self.sin
        self.train['holiday'] = self.doniti
        self.test['hour_cos'] = self.cos
        self.test['hour_sin'] = self.sin
        self.test['holiday'] = self.doniti

        pprint.pprint(self.train)
        with open('data/pickle/train_'+str(distance)+'.pickle', 'wb') as f:
            pickle.dump(self.train,f)
        with open('data/pickle/test_'+str(distance)+'.pickle', 'wb') as f:
            pickle.dump(self.test,f)

if __name__=='__main__':
    my=timeadd()