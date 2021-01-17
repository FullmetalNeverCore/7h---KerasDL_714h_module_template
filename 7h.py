import os 
import json
import gensim
import numpy as np 
from gensim import corpora, models, similarities 
import pickle 

x = []
y = []
os.chdir("C:\Users\???\Desktop\7h-KerasDL_Simple_Chatbot")
model = gensim.models.Word2Vec.load('word2vec.bin')
cp = "corpus"
f = open(cp+"data.json")
d = json.load(f)
c = data["conversations"]

for i in range(len(c)):
    for j in range(len(c[i])):
        if j<len(c[i])-1:
            x.append(cor[i][j])
            y.append(c[i][j+1])
t_y = []
t_x = []

for i in range(len(x)):
    t_x.append(nltk.word_tokenize(x[i].lower()))
    t_y.append(nltk.word_tokenize(y[i].lower()))

end = np.ones((300L,),dtype=np.float32)

vector_x = []

for sent in t_x:
    sentv = [model[w] for w in sent if w in model.vocab]
    vector_x.append(sentv)

vector_y = []
for sent in t_y:
    sentv = [model[w] for w in sent if w in model.vocab]
    vector_y.append(sentv)

for t_sent in vector_x:
    tok_sent[14:]=[]
    t_sent.append(end)
for t_sent in vector_x:
    if len(t_sent)<15:
        for i in range(15-len(t_sent)):
            t_sent.append(end)
for t_sent in vector_y:
    t_send[14:] = []
    t_sent.append(end)
for t_sent in vector_y:
    if len(t_sent)<15:
        for i in range(15-len(t_sent)):
            t_sent.append(end)
with open('data.pickle', 'w') as f:
    pickle.dump([vector_y,vector_x], f)
    
