#coding:utf-8
#地理情報から各訓練データを作成するクラス

import pandas as pd
import numpy as np
from calc_distance import pt_distance

class point_datacreate():
    def __init__(self):
        self.temp = pd.read_csv('data/fixed_temperature.csv',na_values='-')
        self.prec = pd.read_csv('data/fixed_precipitation.csv',na_values='-')
        self.wind = pd.read_csv('data/fixed_wind.csv',na_values='-')
        self.mwind = pd.read_csv('data/fixed_wind_max.csv',na_values='-')
        self.train = pd.read_csv('data/train.csv')
        self.max_distance = 20

        self.temp = self.temp.fillna(method='ffill')
        self.prec = self.prec.fillna(method='ffill')
        self.wind = self.wind.fillna(method='ffill')
        self.mwind = self.mwind.fillna(method='ffill')
    def set_max_distance(self,dist):
        self.max_distance = dist
    #特定の範囲内の観測地点のidを返す関数
    def get_point_id(self,n,data):
        ids = data.drop_duplicates(['局ID'])['局ID']
        ids = [int(u) for u in ids]
        # 最大距離内の観測地点ID
        myids = []
        for i in ids:
            if pt_distance(n, i) < self.max_distance:
                myids.append(i)
        return myids
    #観測地点IDと各種データから平均化されたデータを返す関数
    def get_near_data_mean(self,data,ids):
        df=pd.DataFrame()
        for i in ids:
            df[str(i)]=data[data['局ID'].isin([i])]['測定値'].as_matrix()
        meandata=[]
        for i in range(len(df)):
            meandata.append(np.mean([float(u) for u in df.ix[i].tolist()]))
        return meandata
    # 観測地点IDと各種データから最大値のデータを返す関数
    def get_near_data_max(self, data, ids):
        df = pd.DataFrame()
        for i in ids:
            df[str(i)] = data[data['局ID'].isin([i])]['測定値'].as_matrix()
        meandata = []
        for i in range(len(df)):
            meandata.append(np.max([float(u) for u in df.ix[i].tolist()]))
        return meandata
    #各ファイルから局IDを取得
    def get_distance(self):
        name=['tyuou','keihintohoku','keiyou','saikyoukawagoe','sotobou','syonan','takasaki','uchibou','utsunomiya','yamanote']
        for n in name:
            df=pd.DataFrame()
            df['state']=self.train[n].tolist()
            ids=self.get_point_id(n,self.temp)
            df['temp']=self.get_near_data_mean(self.temp,ids)
            ids = self.get_point_id(n, self.prec)
            df['prec'] = self.get_near_data_mean(self.prec, ids)
            ids = self.get_point_id(n, self.wind)
            df['wind'] = self.get_near_data_mean(self.wind, ids)
            ids = self.get_point_id(n, self.mwind)
            df['mwind'] = self.get_near_data_mean(self.mwind, ids)
            df.to_csv('data/point_train/'+n+'_train.csv',index=None)


if __name__=='__main__':
    my=point_datacreate()
    my.get_distance()