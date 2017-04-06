#coding:utf-8
#よそくするお

from random_forest import RandomForestC
import pandas as pd

if __name__=='__main__':
    my=RandomForestC()
    my.set_trees(1)
    my.model_create()
    my.fit(my.data.ix[:,6:],my.data[['tyuou','keihintohoku','keiyou','uchibou','saikyoukawagoe']])
    result=my.predict(my.data.ix[:,6:])
    data=pd.read_csv('data/sample_submit.csv',names=['name','1','2','3','4'],dtype={'name':str})
    submit=pd.DataFrame()
    submit['name']=data['name'].tolist()
    result.extend(result)
    result.extend(result)
    submit['1']=result[0]
    submit['2'] = result[1]
    submit['3'] = result[2]
    submit['4'] = result[3]
    submit.to_csv('submit.csv',index=False,header=False)
