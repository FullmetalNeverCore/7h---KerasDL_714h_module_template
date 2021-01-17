import os 
from scipy import spatial 
import numpy as np 
import gensim 
import nltk 
from keras.models import load_model 

import theano

theano.config.optimizer="None"

model = load_model("LSTM5000.h5")

modz = gensim.model.Word2Vec.load('word2vec.bin')

while True:
    x=input("Say something:")
    end=np.ones((300L,),dtype=np.float32)
    sent=nltk.word_tokenize(x.lower())
    svector = [modz[w]for w in sent if w in modz.vocab]
    svector[14:]=[]
    svector.append(end)
    if len(svector)<15:
        for i in range(15-len(svector)):
            svector.append(end)
    svector=np.array([svector])
    prediction = model.predict(svector)
    outlist = [mod.most_siliar([prediction[0][i]])[0][0] for i in range(15)]
    out =' '.join(outlist)
    print(out)