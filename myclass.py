#coding:utf-8
#他のクラスの親となる基底クラス

import pandas as pd


class myclass:
    def __init__(self):
        self.dt=self.read_csv('detail')
        self.ct=self.read_csv('train')
        self.ref=self.read_tsv('reference')
        self.detail=self.get_detail_acc()
    #csvデータを読み込み，DF型で返す関数
    def read_csv(self,name,header=None):
        if header is None:
            df = pd.read_csv("data/"+name+".csv")
        else:
            df = pd.read_csv("data/"+name+".csv",names=header)
        return df
    #tsvデータを読み込み，DF型で返す関数
    def read_tsv(self, name,header=None):
        if header is None:
            df = pd.read_csv("data/" + name + ".tsv", delimiter='\t',dtype={'detail_name':str,'target_name':str})
        else:
            df = pd.read_csv("data/" + name + ".tsv", delimiter='\t', dtype={'detail_name': str, 'target_name': str},names=header)
        return df
    #詳細データから個数の配列を返す関数
    def get_detail_acc(self):
        col=self.dt.columns[1:]
        dic={}
        for i in col:
            tmp=self.dt[i].value_counts().to_dict()
            for j in tmp.keys():
                if str(j) in dic:
                    dic[str(j)] =dic[str(j)] + tmp[j]
                else:
                    dic[str(j)] = tmp[j]
        for j in dic.keys():
            dic[j]=[dic[j]]
        return pd.DataFrame.from_dict(dic)
    #id検索をしてリファレンスから返す関数
    def get_ref(self,id,detail):
        if detail:
            return self.ref[self.ref['detail_id'].isin([id])]['detail_name'].get(int(id))
        else:
            return self.ref[self.ref['detail_id'].isin([id])]['target_name'].get(int(id))