#coding:utf-8
#双方向LSTMで予測するモデル

from RNN_predict import RNN_predict
from keras.models import Sequential
from keras.layers.core import Dense, Activation,Dropout
from keras.layers import Activation,Merge,TimeDistributed
from keras.layers.recurrent import LSTM
import pickle
import pandas as pd
import numpy as np

def fork (model, n=2):
    forks = []
    for i in range(n):
        f = Sequential()
        f.add (model)
        forks.append(f)
    return forks

class BIO_predict(RNN_predict):
    def model_create(self):
        left = Sequential()
        left.add(LSTM(output_dim=24, init='uniform', inner_init='uniform',
               forget_bias_init='one', return_sequences=True, activation='tanh',
               inner_activation='sigmoid', input_shape=(len(self.train_x[0]),len(self.train_x[0,0]))))
        right = Sequential()
        right.add(LSTM(output_dim=24, init='uniform', inner_init='uniform',
                      forget_bias_init='one', return_sequences=True, activation='tanh',
                      inner_activation='sigmoid', input_shape=(len(self.train_x[0]), len(self.train_x[0, 0])),go_backwards=True))
        model = Sequential()
        model.add(Merge([left, right], mode='sum'))
        left, right = fork(model)
        left.add(LSTM(output_dim=4, init='uniform', inner_init='uniform',
                      forget_bias_init='one', return_sequences=False, activation='tanh',
                      inner_activation='sigmoid', input_shape=(len(self.train_x[0]), len(self.train_x[0, 0])),
                      dropout=0.5, recurrent_dropout=0.5))
        right.add(LSTM(output_dim=4, init='uniform', inner_init='uniform',
                       forget_bias_init='one', return_sequences=False, activation='tanh',
                       inner_activation='sigmoid', input_shape=(len(self.train_x[0]), len(self.train_x[0, 0])),
                       dropout=0.5, recurrent_dropout=0.5, go_backwards=True))


        self.model = Sequential()
        self.model.add(Merge([left, right], mode='sum'))
        #self.model.add(Merge([left2, right2], mode='sum'))
        self.model.add(Dense(4))
        self.model.add(Activation('softmax'))
        self.model.compile(loss="categorical_crossentropy", optimizer="adam")
    def fit(self):
        self.model.fit([self.train_x,self.train_x],self.train_y,validation_split=0,verbose=2)
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
        submit.to_csv('lstm_submit.csv', index=False, header=False, float_format='%.10f')

    def predict(self):
        pred = self.model.predict_proba([self.test_x,self.test_x], verbose=2)

if __name__=='__main__':
    my=BIO_predict(18)
    my.submit()