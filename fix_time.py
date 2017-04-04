#coding:utf-8
#データの日時を修正するクラス

from myclass import myclass
import pandas as pd
import datetime

def fixtime(name):
    df=pd.read_csv('data/'+name+'.csv',names=['観測日時','局ID','市町村区コード','データ種別コード','品質コード','測定値'],parse_dates=['観測日時'])
    print(df)
    ntime=df['観測日時'].ix[0]
    print(ntime)
    if ntime != datetime.datetime(year=2012,month=1,day=1,hour=0,minute=0):
        print('最初の日付が違います')
        exit(9999)
    fixed=[False]
    for i in range(1,78768):
        if df['観測日時'].ix[i] == ntime+datetime.timedelta(minutes=10):
            fixed.append(False)
        else:
            tmp=df.ix[i-1]
            tmp['観測日時']=ntime+datetime.timedelta(minutes=10)
            print(tmp)
            print(df.columns)
            df.insert(i,df.columns,tmp)
            fixed.append(True)
            print('Fixed!')
        ntime=df['観測日時'].ix[i]
    print(len(df==True))

if __name__=='__main__':
    fixtime('temperature')