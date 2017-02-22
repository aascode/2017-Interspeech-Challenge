# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 09:59:52 2017

@author: HGY
"""

#import keras
import theano
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
import numpy as np
import pickle
import scipy.io
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

FEA_ROOT = '../features/ComPare_2016/'
MIXURE = 256

def getScore(y_true, y_predict):
    y_true = y_true.argmax(1)
    y_predict = y_predict.argmax(1)
    uar = recall_score(y_true, y_predict, average='macro')  
    accu = accuracy_score(y_true, y_predict)  
    return uar, accu

### ----------------------------  Load traiing data & devel data (fisher-encoded)  --------------------------------
# Laod train Labels
with open('../lab/label_train.pickle', 'rb') as handle:
    content = pickle.load(handle)
content = sorted(content.items())
Label_train = np.array([x[1] for x in content])

# Load train data
mat = scipy.io.loadmat(FEA_ROOT+'FV_train_m'+str(MIXURE)+'.mat')
Data_train = mat['FV_train']

# Laod devel Labels
with open('../lab/label_devel.pickle', 'rb') as handle:
    content = pickle.load(handle)
content = sorted(content.items())
Label_devel= np.array([x[1] for x in content])

# Load devel data
mat = scipy.io.loadmat(FEA_ROOT+'FV_devel_m'+str(MIXURE)+'.mat')
Data_devel = mat['FV_devel']


### ----------------------------  Preprocess data  --------------------------------
Label_train  = np_utils.to_categorical(Label_train).astype('float32')
Label_devel = np_utils.to_categorical(Label_devel).astype('float32')
Data_train = Data_train[:,1:].astype('float32')
Data_devel = Data_devel[:,1:].astype('float32')




### ----------------------------  DNN  --------------------------------
# 宣告這是一個 Sequential 次序性的深度學習模型
model = Sequential()

# 加入第一層 hidden layer (128 neurons)
# [重要] 因為第一層 hidden layer 需連接 input vector,故需要在此指定 input_dim
model.add(Dense(128, input_dim=262*MIXURE, activation='sigmoid', name='1st hidden'))
model.add(Dense(256, activation='sigmoid', name='2nd hidden'))
model.add(Dense(2, activation='softmax', name='output'))


# 觀察 model summary
model.summary()

# 指定 loss function 和 optimizier
sgd = SGD(lr=0.01,momentum=0.0,decay=0.0,nesterov=False)
adam = Adam(lr=0.01)
model.compile(loss='binary_crossentropy',optimizer=sgd, metrics=['accuracy'])


# 指定 batch_size, nb_epoch, validation 後，開始訓練模型!!!
history = model.fit(Data_train, Label_train,
                    batch_size=32,
                    verbose=1,
                    nb_epoch=30,
                    validation_data = (Data_devel, Label_devel))
Predict_devel = model.predict(Data_devel, batch_size=32, verbose=1)
weight = model.get_weights()


### ----------------------------  Report  --------------------------------
# list all data in history
#print(history.history.keys()) # list contents in the history
uar_devel, accu_devel = getScore(Label_devel, Predict_devel)
print 'UAR:', uar_devel, 'Accuracy:', accu_devel            # Keras binary_accuracy = accuracy

# plot
plt.figure()
p1 = plt.subplot(121)
p2 = plt.subplot(122)

p1.plot(history.history['accuracy'])
p1.plot(history.history['accuracy'])
p2.plot(history.history['loss'])
p2.plot(history.history['val_loss'])

p1.set_title('Model Accuracy')
p1.set_ylabel('Accuracy')
p1.set_xlabel('Epoches')
p1.legend(['train', 'devel'], loc='upper right')

p2.set_title('Model Loss')
p2.set_ylabel('loss')
p2.set_xlabel('Epoches')
p2.legend(['train', 'devel'], loc='upper right')
plt.show()
