# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(2016)

import os
import glob
import math
import pickle
import datetime

from keras.layers import Input, Embedding, LSTM, Dense,Flatten, Dropout, merge
from keras.models import Model

def load_train():
    X_train_uid=[]
    X_train_iid=[]
    Y_train_score=[]

    path = os.path.join('./data',  'train.csv')
    print('Read train data',path)

    f = open(path, 'r')
    line = f.readline()
    while (1):
        line = f.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        X_train_uid.append(int(arr[0]))
        X_train_iid.append(int(arr[1]))
        Y_train_score.append(int(arr[2]))
    f.close()
    return X_train_uid,X_train_iid,Y_train_score

def load_test():
    X_test_uid=[]
    X_test_iid=[]

    path = os.path.join('./data',  'test.csv')
    print('Read test data',path)

    f = open(path, 'r')
    line = f.readline()
    while (1):
        line = f.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        X_test_uid.append(int(arr[0]))
        X_test_iid.append(int(arr[1]))
    f.close()
    return X_test_uid,X_test_iid


X_train_uid,X_train_iid,Y_train_score = load_train()
#print len(X_train_uid),X_train_uid[33177260],max(X_train_uid)
#print len(X_train_iid),X_train_iid[33177260],max(X_train_iid)
#print len(Y_train_score),Y_train_score[33177260]
print "load train data OK."

X_test_uid,X_test_iid = load_test()
#print len(X_test_uid),X_test_uid[100],max(X_test_uid)
#print len(X_test_iid),X_test_iid[100],max(X_test_iid)
print "load test data OK."

# normalize train date
X_train_uid=np.array(X_train_uid)
X_train_uid=X_train_uid.reshape(X_train_uid.shape[0],1)

X_train_iid=np.array(X_train_iid)
X_train_iid=X_train_iid.reshape(X_train_iid.shape[0],1)

Y_train_score = np.array(Y_train_score).astype('float32')
Y_train_score = (Y_train_score - 1)/ 4

# normalize test date
X_test_uid=np.array(X_test_uid)
X_test_uid=X_test_uid.reshape(X_test_uid.shape[0],1)

X_test_iid=np.array(X_test_iid)
X_test_iid=X_test_iid.reshape(X_test_iid.shape[0],1)

# define model
input_1=Input(shape=(1,), dtype='int32')
input_2=Input(shape=(1,), dtype='int32')
x1=Embedding(output_dim=128, input_dim=223970, input_length=1)(input_1)
x2=Embedding(output_dim=128, input_dim=14726, input_length=1)(input_2)
x1=Flatten()(x1)
x2=Flatten()(x2)
x = merge([x1, x2], mode='concat')
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
out = Dense(1, activation='sigmoid')(x)
model = Model(input=[input_1, input_2], output=out)
model.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=[])
# train model
model.fit([X_train_uid, X_train_iid], Y_train_score,
          nb_epoch=10, batch_size=1024*6)

# predict
Y_test_score = model.predict([X_test_uid, X_test_iid],batch_size=2048)
Y_test_score = Y_test_score * 4 + 1

f=open("out.csv","w")
f.write("score\n")
for i in range(Y_test_score.shape[0]):
    f.write("{:1.4f}".format(Y_test_score[i,0]))
    f.write("\n")
f.close()

