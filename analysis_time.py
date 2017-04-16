#coding:utf-8
#データ時間相関分析

import pandas as pd
import tqdm
import pickle
import seaborn as sns
import datetime

class analysis_time():
    def __init__(self):
        self.read_data()
    def read_data(self,create=False):
        if create:
            self.df = pd.read_csv('data/train.csv', parse_dates=True, index_col=0)
        else:
            with open('data/hour_data.pickle','rb') as f:
                self.hour = pickle.load(f)
            with open('data/days_data.pickle','rb') as f:
                self.days = pickle.load(f)
            with open('data/month_data.pickle','rb') as f:
                self.month = pickle.load(f)
            #データを可視化用に変換
            tmp=[]
            for i in range(24):
                for j in range(1,4):
                    if j==1:
                        name='people'
                    elif j==2:
                        name='machine'
                    else:
                        name='weather'
                    tmp.append([str(i+1),self.hour[i][j],name])
            self.v_hour = pd.DataFrame(tmp,columns=['hour','acc_num','category'])

            tmp = []
            for i in range(7):
                for j in range(1, 4):
                    if j == 1:
                        name = 'people'
                    elif j == 2:
                        name = 'machine'
                    else:
                        name = 'weather'
                    tmp.append([self.fromweekdaytoname(i), self.days[i][j], name])
            self.v_days = pd.DataFrame(tmp, columns=['days', 'acc_num', 'category'])

            tmp = []
            for i in range(12):
                for j in range(1, 4):
                    if j == 1:
                        name = 'people'
                    elif j == 2:
                        name = 'machine'
                    else:
                        name = 'weather'
                    tmp.append([i+1, self.month[i][j], name])
            self.v_month = pd.DataFrame(tmp, columns=['month', 'acc_num', 'category'])
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
    def view_data(self):
        myorder=[]
        for i in range(1,25):
            myorder.append(str(i))
        ax = sns.barplot(x='hour',y='acc_num',hue='category',order=myorder,data=self.v_hour)
        sns.plt.title('hour vs num of accidents')
        sns.plt.show()

        ax = sns.barplot(x='days', y='acc_num', hue='category', data=self.v_days)
        sns.plt.title('weekday vs num of accidents')
        sns.plt.show()

        ax = sns.barplot(x='month', y='acc_num', hue='category', data=self.v_month)
        sns.plt.title('month vs num of accidents')
        sns.plt.show()
    def fromweekdaytoname(self,num):
        if num==0:
            return 'monday'
        elif num==1:
            return 'tuesday'
        elif num==2:
            return 'wendesday'
        elif num==3:
            return 'thursday'
        elif num==4:
            return 'friday'
        elif num==5:
            return 'saturday'
        else:
            return 'sunday'



if __name__=='__main__':
    my=analysis_time()
    my.view_data()