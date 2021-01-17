import os 
import json
import gensim
import numpy as np 
from gensim import corpora, models, similarities 
import pickle 
from keras.layers.recurrent import LSTM,SimpleRNN
from sklearn.model_selection import train_test_split
import theano 

theano.config.optimizer = "None"

with open('data.pickle') as d:
    vector_x,vector_y=pickle.load(d) 

vector_x=np.array(vector_x,dtype=np.float64)
vector_y=np.array(vector_y,dtype=np.float64)

x_tra,x_ts, y_tra, y_ts = train_test_split(vector_x, vector_y, test_size=0.2, random_state=1)

model = Sequential()

for x in range(4):
    model.add(LSTM(output_dim=300, input_shape=x_train.shape[1:], return_sequential=True, init='glorot_normal', inner_init='glorot_normal', activation='sigmoid'))
model.compile(loss='cosine_proximity', optimizer='adam', metrics=['accuaracy'])

model.fit(x_tra, y_tra, nb_epoch=500, validation_data=(x_ts, y_ts))
model.save('LSTM500.h5')
model.fit(x_tra, y_tra, nb_epoch=500, validation_data=(x_ts, y_ts))
model.save('LSTM1000.h5')
model.fit(x_tra, y_tra, nb_epoch=500, validation_data=(x_ts, y_ts))
model.save('LSTM1500.h5')
model.fit(x_tra, y_tra, nb_epoch=500, validation_data=(x_ts, y_ts))
model.save('LSTM2000.h5')
model.fit(x_tra, y_tra, nb_epoch=500, validation_data=(x_ts, y_ts))
model.save('LSTM2500.h5')
model.fit(x_tra, y_tra, nb_epoch=500, validation_data=(x_ts, y_ts))
model.save('LSTM3000.h5')
model.fit(x_tra, y_tra, nb_epoch=500, validation_data=(x_ts, y_ts))
model.save('LSTM3500.h5')
model.fit(x_tra, y_tra, nb_epoch=500, validation_data=(x_ts, y_ts))
model.save('LSTM4000.h5')
model.fit(x_tra, y_tra, nb_epoch=500, validation_data=(x_ts, y_ts))
model.save('LSTM4500.h5')
model.fit(x_tra, y_tra, nb_epoch=500, validation_data=(x_ts, y_ts))
model.save('LSTM5000.h5')
prediction=model.predict(x_ts)
md = genshim.models.Word2Vec.load('word2vec.bin')
[mod.most_similar([prediction[10][i]])[0] for i in range(15)]