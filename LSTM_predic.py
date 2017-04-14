#coding:utf-8
#LSTMで予測するモデル

from RNN_predict import RNN_predict
from keras.models import Sequential
from keras.layers.core import Dense, Activation,Dropout
from keras.layers.recurrent import LSTM
import pickle
import pandas as pd
import numpy as np

class LSTM_predict(RNN_predict):
    def model_create(self):
        self.model = Sequential()
        self.model.add(LSTM(12,batch_input_shape=(None, len(self.train_x[0]),len(self.train_x[0,0]))))
        self.model.add(Activation("relu"))
        self.model.add(Dense(8, input_dim=12))
        self.model.add(Activation("relu"))
        self.model.add(Dense(4, input_dim=8))
        self.model.add(Activation("softmax"))
        self.model.compile(loss="categorical_crossentropy", optimizer="adam")
    def save_submit(self,pred):
        data = pd.read_csv('data/sample_submit.csv', names=['name', '1', '2', '3', '4'], dtype={'name': str})
        submit=pd.DataFrame(data['name'],columns=['name'])
        #時系列入力のため最初に補間が必要
        static_pred=[0.96033,0.01916,0.01143,0.00908]
        for i in range(4):
            time_num=len(submit) - len(pred)
            my_insert=[static_pred[i]] * time_num
            submit[str(i)]=np.hstack([my_insert,pred[:,i]])
        submit.to_csv('lstm_submit.csv', index=False, header=False, float_format='%.10f')

if __name__=='__main__':
    my=RNN_predict()
    my.submit()