#coding:utf-8
#データを結合するクラス

from myclass import myclass

class data_manging(myclass):
    def __init__(self):
        #各データを読み込む
        self.train=self.read_csv('train')

if __name__=='__main__':
    my=data_manging()