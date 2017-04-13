#coding:utf-8
#ニューラルネットで予測するモデル

from neuralnet import neuralnetC
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation,Dropout
import pickle

class predict_neuralnet(neuralnetC):
    def read_data(self):
        self.train = pd.DataFrame()
        self.test=pd.DataFrame()
        self.train=pd.concat([self.train,pd.read_csv('data/point_train/tyuou_' + str(self.maxdistance) + '_train.csv')],ignore_index=True)
        self.train =self.train.append(pd.read_csv('data/point_train/keihintohoku_' + str(self.maxdistance) + '_train.csv'))
        self.train =self.train.append(pd.read_csv('data/point_train/keiyou_' + str(self.maxdistance) + '_train.csv'))
        self.train =self.train.append(pd.read_csv('data/point_train/uchibou_' + str(self.maxdistance) + '_train.csv'))
        self.train = self.train.append(pd.read_csv('data/point_train/saikyoukawagoe_' + str(self.maxdistance) + '_train.csv'))
        self.test=pd.concat([self.test,pd.read_csv('data/point_train/sotobou_' + str(self.maxdistance) + '_test.csv')],ignore_index=True)
        self.test=self.test.append(pd.read_csv('data/point_train/utsunomiya_' + str(self.maxdistance) + '_test.csv'))
        self.test =self.test.append(pd.read_csv('data/point_train/yamanote_' + str(self.maxdistance) + '_test.csv'))
        self.test =self.test.append(pd.read_csv('data/point_train/syonan_' + str(self.maxdistance) + '_test.csv'))
        self.test =self.test.append(pd.read_csv('data/point_train/takasaki_' + str(self.maxdistance) + '_test.csv'))
        self.alldata=pd.concat([self.train,self.test],ignore_index=True)
    def fit(self,x_train,y_train):
        self.train=self.zscore(self.train)
        self.model.fit(x_train,y_train,validation_split=0.1,verbose=2)
    def predict(self,x_test):
        #pred=self.model.predict_proba(x_test)
        pred = self.model.predict_proba(x_test,verbose=1)
        return pred
    def submit(self):
        train=self.zscore(self.train)
        self.model_create()
        self.fit(train.ix[:,4:].as_matrix(),train.ix[:,:4].as_matrix())
        test=self.zscore(self.test)
        pred=self.predict(test.as_matrix())
        self.save_submit(pred)
    def save_submit(self,pred):
        data = pd.read_csv('data/sample_submit.csv', names=['name', '1', '2', '3', '4'], dtype={'name': str})
        submit=pd.DataFrame(data['name'],columns=['name'])
        for i in range(4):
                submit[str(i)]=pred[:,i]
        submit.to_csv('nn_submit.csv', index=False, header=False, float_format='%.10f')
    def model_create(self):
        input_length=len(self.use_var)
        self.model = Sequential()
        self.model.add(Dense(input_length, input_dim=input_length))
        self.model.add(Activation("relu"))
        self.model.add(Dense(input_length, input_dim=input_length))
        self.model.add(Activation("relu"))
        self.model.add(Dense(4, input_dim=input_length))
        self.model.add(Activation("softmax"))
        self.model.compile(loss="categorical_crossentropy", optimizer="adam")
    def set_var(self,var):
        self.use_var=var
    def pickle_data(self):
        train = self.zscore(self.train)
        test = self.zscore(self.test)
        with open('data/pickle/train.pickle','wb') as f:
            pickle.dump(train, f)
        with open('data/pickle/test.pickle','wb') as f:
            pickle.dump(test, f)




if __name__=='__main__':
    my=predict_neuralnet()
    my.set_var(['temp','prec','wind','mwind'])
    #my.set_distance(20)
    my.read_data()
    my.pickle_data()
    #my.submit()