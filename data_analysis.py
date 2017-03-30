#coding:utf-8
#各データの形を見る

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter


class deta_analysis:
    def __init__(self):
        self.dt=self.read_csv('detail')
        self.ct=self.read_csv('train')
        self.ref=self.read_tsv('reference')
        self.detail=self.get_detail_acc()
    #csvデータを読み込み，DF型で返す関数
    def read_csv(self,name):
        df = pd.read_csv("data/"+name+".csv")
        return df
    #tsvデータを読み込み，DF型で返す関数
    def read_tsv(self, name):
        df = pd.read_csv("data/" + name + ".tsv", delimiter='\t',dtype={'detail_name':str,'target_name':str})
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
    #詳細障害の棒グラフを表示する関数(acc:Trueで障害なし表示)
    def show_circle_dt(self,acc):
        labels=[]
        nums=[]
        for i in self.detail.columns:
            labels.append(self.get_ref(int(i),True))
            nums.append(self.detail[i].ix[0])
        #２つのリストをzipでまとめてソートする
        tmp=list(zip(labels,nums))
        tmp.sort(key=itemgetter(1),reverse=True)
        labels,nums=zip(*tmp)
        sns.set(font='TakaoExMincho')
        if acc:
            ax=sns.barplot(y=labels,x=nums)
        else:
            ax=sns.barplot(y=labels[1:], x=nums[1:])
        total = 0
        for p in ax.patches:
            total += p.get_width()
        for p in ax.patches:
            width = p.get_width()
            ax.text(width + total / 50,
                    p.get_y() + p.get_height() / 2+0.1,
                    '{:1.2f}%'.format(width / total * 100),
                    ha="center")
        plt.title('各障害の総発生時間(詳細)')
        plt.xlabel('時間(10分)')
        plt.show()
    #カテゴリ障害の棒グラフを表示する関数(acc:Trueで障害なし表示)
    def show_circle_ct(self, acc):
        dic={'異常なし':0,'人身支障':0,'機械支障':0,'気象支障':0}
        for i in self.detail.columns:
            dic[self.get_ref(int(i), False)]+=self.detail[i].ix[0]
        # ２つのリストをzipでまとめてソートする
        tmp = list(zip(dic.keys(), dic.values()))
        tmp.sort(key=itemgetter(1), reverse=True)
        labels, nums = zip(*tmp)
        sns.set(font='TakaoExMincho')
        if acc:
            ax=sns.barplot(y=labels, x=nums)
        else:
            ax=sns.barplot(y=labels[1:], x=nums[1:])
        total=0
        for p in ax.patches:
            total+=p.get_width()
        for p in ax.patches:
            width=p.get_width()
            ax.text(width + total/30,
                    p.get_y() + p.get_height() / 2.,
                    '{:1.3f}%'.format(width / total*100),
                    ha="center")
        plt.title('各障害の総発生時間')
        plt.xlabel('時間(10分)')
        plt.show()
    #カテゴリ障害の平均時間を求める関数
    def calc_acctime_ct(self):
        isacc=False
        occur={'異常なし': 0, '人身支障': 0, '機械支障': 0, '気象支障': 0}
        dic = {'異常なし': 0, '人身支障': 0, '機械支障': 0, '気象支障': 0}
        for i in self.detail.columns:
            dic[self.get_ref(int(i), False)] += self.detail[i].ix[0]
        for i in range(1,6):
            for j in range(len(self.dt)):
                if isacc==False:
                    if self.dt.ix[j,i] != 0:
                        occur[self.get_ref(int(self.dt.ix[j,i]),False)] += 1
                        isacc=True
                else:
                    if self.dt.ix[j,i] == 0:
                        isacc=False
        a=np.array(list(dic.values()))
        b=np.array(list(occur.values()))
        result=1.0*a[1:]/b[1:]
        sns.set(font='TakaoExMincho')
        sns.barplot(y=list(dic.keys())[1:],x=result)
        plt.title('各障害の平均発生時間')
        plt.xlabel('時間(10分)')
        plt.show()
    # 詳細障害の平均時間を求める関数
    def calc_acctime_dt(self):
        isacc = False
        occur={}
        labels = []
        nums = []
        for i in self.detail.columns:
            labels.append(self.get_ref(int(i), True))
            nums.append(self.detail[i].ix[0])
            occur[self.get_ref(int(i), True)]=0
        for i in range(1, 6):
            for j in range(len(self.dt)):
                if isacc == False:
                    if self.dt.ix[j, i] != 0:
                        occur[self.get_ref(int(self.dt.ix[j, i]), True)] += 1
                        isacc = True
                else:
                    if self.dt.ix[j, i] == 0:
                        isacc = False
        nums=1.0*np.array(nums)/list(occur.values())
        # ２つのリストをzipでまとめてソートする
        tmp = list(zip(labels, nums))
        tmp.sort(key=itemgetter(1), reverse=True)
        labels, nums = zip(*tmp)
        sns.set(font='TakaoExMincho')
        sns.barplot(y=labels[1:], x=nums[1:])
        plt.title('各障害の平均発生時間')
        plt.xlabel('時間(10分)')
        plt.show()


if __name__=='__main__':
    my=deta_analysis()
    my.show_circle_dt(acc=False)