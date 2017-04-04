#coding:utf-8
#データを結合するクラス

import pandas as pd
import numpy as np
from myclass import myclass

class data_manging(myclass):
    def __init__(self):
        #各データを読み込む
        self.raw_train=self.read_csv('train')
        self.temperature=pd.read_csv('data/temperature.csv',names=['観測日時','局ID','市町村区コード','データ種別コード','品質コード','測定値'],na_values='-')
        #欠損値を前方穴埋めで置換
        self.temperature=self.temperature.fillna(method='ffill')
        self.point=self.read_tsv('observation_point')
        self.create_train()
        print(self.temperature.isnull().any(axis=0))
        print(len(self.train))
    #訓練データの作成
    def create_train(self):
        #とりあえず中央線のみ
        self.train=pd.DataFrame(data=self.raw_train['tyuou'].as_matrix(),columns=['predict'])
    #特定の地点のデータを抜き出し,DF型で返す関数
    def get_feature(self,data,point_id):
        return data[data['局ID'].isin([point_id])]
    #各観測地点のデータを説明変数に追加していく関数
    def add_feature(self,data,name):
        for i in self.point['局ID']:
            print(len(self.get_feature(data,i)['測定値']))
            self.train[name+str(i)]=self.get_feature(data,i)['測定値']
        print(self.train)


if __name__=='__main__':
    my=data_manging()
    my.add_feature(my.temperature,'temp')