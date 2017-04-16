#coding:utf-8
#データ時間相関分析

import pandas as pd
import datetime
import tqdm
import pickle

class analysis_time():
    def __init__(self):
        self.df = pd.read_csv('data/train.csv',parse_dates=True,index_col=0)
    def read_data(self):
        pass
    def hour_data(self):
        #1h毎の発生頻度データ
        hour=[]
        for i in range(24):
            hour.append([0,0,0,0])
        count=0
        for i in tqdm.tqdm(self.df.index):
            for j in range(5):
                hour[i.hour][self.df.ix[count,j]] += 1
            count+=1
        print(hour)
        with open('data/hour_data.pickle','wb') as f:
            pickle.dump(hour, f)
    def weekday_data(self):
        #曜日毎の発生頻度データ
        days=[]
        for i in range(7):
            days.append([0,0,0,0])
        count=0
        for i in tqdm.tqdm(self.df.index):
            for j in range(5):
                days[i.weekday()][self.df.ix[count,j]] += 1
            count+=1
        print(days)
        with open('data/days_data.pickle','wb') as f:
            pickle.dump(days, f)
    def month_data(self):
        # 月毎の発生頻度データ
        month = []
        for i in range(12):
            month.append([0, 0, 0, 0])
        count = 0
        for i in tqdm.tqdm(self.df.index):
            for j in range(5):
                month[i.month-1][self.df.ix[count, j]] += 1
            count += 1
        print(month)
        with open('data/month_data.pickle', 'wb') as f:
            pickle.dump(month, f)

if __name__=='__main__':
    my=analysis_time()
    my.month_data()