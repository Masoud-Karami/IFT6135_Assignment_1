#!/usr/bin/env python
# coding: utf-8

# In[44]:


import numpy as np
import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda, Flatten, Embedding, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, AvgPool2D
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt


# In[60]:


dim= 28
nclasses = 10
epochs = 10
batch_size = 128
((xtrain, ytrain),
 (xval, yval)) = tf.keras.datasets.mnist.load_data()
xtrain = xtrain/np.float32(255)
ytrain = ytrain.astype(np.int32)  # not required
xval = xval/np.float32(255)
yval = yval.astype(np.int32)  
print(xtrain.shape)
print(xval.shape)


# In[61]:


#def df_reshape(df):
   # print("Previous shape, pixels are in 2D vector:", df.shape)
    #df = np.reshape(df,(-1, dim, dim, 1)) 
    # -1 means the dimension doesn't change
   # print("After reshape, pixels are a 28x28x1 3D matrix:", df.shape)
    #return df


# In[62]:


#xtrain = df_reshape(xtrain)
#xval = df_reshape(xval)


# In[63]:


xtrain = xtrain.reshape(xtrain.shape[0], dim, dim, 1)
xval = xval.reshape(xval.shape[0], dim, dim, 1)
input_shape = (dim, dim, 1)


# In[64]:


#convert class vectors to binary class matrices - this is for use in the categorical_crossentropy loss below
y_train = keras.utils.to_categorical(ytrain, nclasses)
y_val = keras.utils.to_categorical(yval, nclasses)
#print (y_train[10000])


# In[66]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=(dim, dim, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(nclasses, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()

model.fit(xtrain, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(xval, y_val),
          callbacks=[history])
score = model.evaluate(xval, y_val, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(range(1, 11), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()


# In[ ]:




