#coding:utf-8
#データの日時を修正するクラス

from myclass import myclass
import pandas as pd
import datetime
import json

#データの時間抜けを修正
def fixtime(df):
    ntime=df.index[0]
    if ntime != datetime.datetime(year=2012,month=1,day=1,hour=0,minute=0):
        print('最初の日付が違います')
        exit(9999)
    fixed=[False]
    for i in range(1,78768):
        if df.index[i] == ntime+datetime.timedelta(minutes=10):
            fixed.append(False)
        else:
            tmp=pd.DataFrame(columns=df.columns)
            tmp=tmp.append(df.ix[i-1],ignore_index=True)
            tmp.index=[ntime+datetime.timedelta(minutes=10)]
            df=df.append(tmp)
            df=df.sort_index()
            fixed.append(True)
        ntime=df.index[i]
    df['fixed']=fixed

    return df,sum(fixed)

#地点idによってデータを分割
def split_by_id(name):
    point = pd.read_csv("data/observation_point.tsv", delimiter='\t', dtype={'detail_name': str, 'target_name': str})
    df = pd.read_csv('data/' + name + '.csv', names=['観測日時', '局ID', '市町村区コード', 'データ種別コード', '品質コード', '測定値'],parse_dates=True, index_col=0)
    fixed=pd.DataFrame()
    dic={}
    for i in point['局ID']:
        print(str(i)+'\tの処理を実行中')
        tmp,errnum=fixtime(df[df['局ID'].isin([i])])
        fixed=fixed.append(tmp)
        print(fixed)
        dic[str(i)]=errnum

    fixed.to_csv('fixed_'+name+'.csv',index=False)
    jstring=json.dumps(dic)
    with open('output.txt',mode='w') as f:
        f.write(jstring)


if __name__=='__main__':
    split_by_id('temperature')