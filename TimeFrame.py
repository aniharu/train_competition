#coding:utf-8
#時系列入力化するスクリプト

import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm

class create_time_data():
    def __init__(self):
        self.read_pickle()
        self.set_timespan(12)
    def read_pickle(self):
        with open('data/pickle/train_holiday.pickle','rb') as f:
            self.train=pickle.load(f)
        with open('data/pickle/test_holiday.pickle','rb') as f:
            self.test=pickle.load(f)
    def set_timespan(self,span):
        self.span=span
    def create_data(self):
        train_x=[]
        train_y=self.train[['none','people','machine','weather']].ix[self.span:]
        test_x=[]
        for i in tqdm(range(self.span,len(self.train),1)):
            one_step=[]
            for j in range(self.span,0,-1):
                one_step.append(self.train.ix[i-j].tolist()[4:])
            train_x.append(one_step)
        print('\n訓練データ作成完了')
        for i in tqdm(range(self.span,len(self.test),1)):
            one_step=[]
            for j in range(self.span,0,-1):
                one_step.append(self.test.ix[i-j].tolist())
            test_x.append(one_step)
        with open('data/pickle/' + str(self.span) + 'train_holiday_x.pickle', 'wb') as f:
            pickle.dump(np.array(train_x), f)
        with open('data/pickle/' + str(self.span) + 'train_holiday_y.pickle', 'wb') as f:
            pickle.dump(np.array(train_y), f)
        with open('data/pickle/' + str(self.span) + 'test_holiday_x.pickle', 'wb') as f:
            pickle.dump(np.array(test_x), f)



if __name__=='__main__':
    my=create_time_data()
    my.set_timespan(18)
    my.create_data()