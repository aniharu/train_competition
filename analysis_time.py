#coding:utf-8
#データ時間相関分析

import pandas as pd
import datetime
import tqdm

class analysis_time():
    def __init__(self):
        self.df = pd.read_csv('data/train.csv',parse_dates=True,index_col=0)
    def hour_data(self):
        #1h毎の発生頻度用データ
        hour=[]
        for i in range(24):
            hour.append([0,0,0,0])
        count=0
        for i in tqdm.tqdm(self.df.index):
            for j in range(5):
                hour[i.hour][self.df.ix[count,j]] += 1
            count+=1
        print(hour)

if __name__=='__main__':
    my=analysis_time()
    my.hour_data()