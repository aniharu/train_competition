#coding:utf-8
#SVMでためしてみる！

from random_forest import RandomForestC
from sklearn.svm import SVC

class SVM(RandomForestC):
    def __init__(self):
        super().__init__()
    def model_create(self):
        self.model=SVC(verbose=1,random_state=0)

if __name__=='__main__':
    my=SVM()
    my.cross_validation()
