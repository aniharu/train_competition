#coding:utf-8
#RNNで予測するモデル

from neuralnet_predict import predict_neuralnet
from keras.models import Sequential
from keras.layers.core import Dense, Activation,Dropout
from keras.layers.recurrent import SimpleRNN
import pickle
import pandas as pd
import numpy as np

class RNN_predict(predict_neuralnet):
    def __init__(self,span):
        self.set_span(span)
        self.read_data()
    def read_data(self):
        with open('data/pickle/'+str(self.span)+'train_x.pickle','rb') as f:
            self.train_x=pickle.load(f)
        with open('data/pickle/'+str(self.span)+'train_y.pickle','rb') as f:
            self.train_y=pickle.load(f)
        with open('data/pickle/'+str(self.span)+'test_x.pickle','rb') as f:
            self.test_x=pickle.load(f)
        print(len(self.train_x[0,0]))
    def model_create(self):
        self.model = Sequential()
        self.model.add(SimpleRNN(12,batch_input_shape=(None, len(self.train_x[0]),len(self.train_x[0,0]))))
        self.model.add(Activation("relu"))
        #self.model.add(SimpleRNN(8))
        #self.model.add(Activation("relu"))
        self.model.add(Dense(4, input_dim=8))
        self.model.add(Activation("softmax"))
        self.model.compile(loss="categorical_crossentropy", optimizer="adam")
    def fit(self):
        self.model.fit(self.train_x,self.train_y,validation_split=0.1,verbose=2)
    def predict(self):
        pred = self.model.predict_proba(self.test_x,verbose=2)
        return pred
    def submit(self):
        self.model_create()
        self.fit()
        pred=self.predict()
        self.save_submit(pred)
    def save_submit(self,pred):
        data = pd.read_csv('data/sample_submit.csv', names=['name', '1', '2', '3', '4'], dtype={'name': str})
        submit=pd.DataFrame(data['name'],columns=['name'])
        #時系列入力のため最初に補間が必要
        static_pred=[0.96033,0.01916,0.01143,0.00908]
        for i in range(4):
            time_num=len(submit) - len(pred)
            my_insert=[static_pred[i]] * time_num
            submit[str(i)]=np.hstack([my_insert,pred[:,i]])
        submit.to_csv('rnn_submit.csv', index=False, header=False, float_format='%.10f')
    def set_span(self,num):
        self.span=num

if __name__=='__main__':
    my=RNN_predict()
    my.submit()