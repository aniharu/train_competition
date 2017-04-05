#coding:utf-8
#trainデータと説明変数データを結合する関数

import pandas as pd


def data_connect():
    train=pd.read_csv('data/train.csv')
    temp=pd.read_csv('data/fixed_temperature.csv',names=['観測日時', '局ID', '市町村区コード', 'データ種別コード', '品質コード', '測定値','fixed'],na_values='-')
    prec = pd.read_csv('data/fixed_precipitation.csv', names=['観測日時', '局ID', '市町村区コード', 'データ種別コード', '品質コード', '測定値','fixed'],na_values='-')
    wind = pd.read_csv('data/fixed_wind.csv', names=['観測日時', '局ID', '市町村区コード', 'データ種別コード', '品質コード', '測定値','fixed'],na_values='-')
    mwind = pd.read_csv('data/fixed_wind_max.csv', names=['観測日時', '局ID', '市町村区コード', 'データ種別コード', '品質コード', '測定値','fixed'],na_values='-')
    point = pd.read_csv("data/observation_point.tsv", delimiter='\t', dtype={'detail_name': str, 'target_name': str})

    # 欠損値を前方穴埋めで置換
    temp=temp.fillna(method='ffill')
    prec = prec.fillna(method='ffill')
    wind = wind.fillna(method='ffill')
    mwind = mwind.fillna(method='ffill')

    for i in point['局ID']:
        print(str(i)+'\tの処理を実行中')
        tmp=temp[temp['局ID'].isin([i])]['測定値'].as_matrix()
        if len(tmp) == 78768:
            train[str(i)+'_temp']=tmp
        tmp=prec[prec['局ID'].isin([i])]['測定値'].as_matrix()
        if len(tmp) == 78768:
            train[str(i) + '_prec']=tmp
        tmp=wind[wind['局ID'].isin([i])]['測定値'].as_matrix()
        if len(tmp) == 78768:
            train[str(i) + '_wind'] =tmp
        tmp=mwind[mwind['局ID'].isin([i])]['測定値'].as_matrix()
        if len(tmp) == 78768:
            train[str(i) + '_mwind'] = tmp

    train.to_csv('data/connected_train.csv',index=False)


if __name__=='__main__':
    data_connect()