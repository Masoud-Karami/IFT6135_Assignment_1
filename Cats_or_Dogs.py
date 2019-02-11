# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 13:04:41 2019

@author: karm2204
"""

'''
import os

class ImageRename():
    def __init__(self):
        self.path = 'D:/New folder/train'

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)

        i = 10000

        for item in filelist:
            if 'dog' in item:
                src = os.path.join(self.path, item)
                dst = os.path.join(self.path, 'dog.' + str(i) + '.jpg')
                os.rename(src, dst)
                i = i + 1

if __name__ == '__main__':
    newname = ImageRename()
    newname.rename()
'''


import os, shutil

# The path to the directory where the original
# dataset was uncompressed
# trainset is cat and dog all included

original_dataset_dir = 'D:/trainset/all'
# The directory where we will
# store our smaller dataset
base_dir = 'D:/trainset/cat_or_dog'

os.mkdir(base_dir)

img_width = 150
img_height = 150


# we will create a new dataset containing three subsets:
# a training set with 1000 samples of each class, 
# a validation set with 500 samples of each class, and 
# finally a test set with 500 samples of each class.
# Directories for our training,
# validation and test splits


train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

# Directory with our training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)

# Directory with our training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

# Directory with our validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)

# Directory with our validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

# Directory with our validation cat pictures
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)

# Directory with our validation dog pictures
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

# Copy first 2000 out of 9999 cat images to train_cats_dir
fnames = ['{}.Cat.jpg'.format(i) for i in range(1, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

# Copy next 1000 cat images to validation_cats_dir
fnames = ['{}.Cat.jpg'.format(i) for i in range(2000, 3000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy next 1000 cat images to test_cats_dir
fnames = ['{}.Cat.jpg'.format(i) for i in range(3000, 4000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy first 2000 dog images to train_dogs_dir
fnames = ['{}.Dog.jpg'.format(i) for i in range(1, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy next 1000 dog images to validation_dogs_dir
fnames = ['{}.Dog.jpg'.format(i) for i in range(2000, 3000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy next 1000 dog images to test_dogs_dir
fnames = ['{}.Dog.jpg'.format(i) for i in range(3000, 4000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)
    


print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total training dog images:', len(os.listdir(train_dogs_dir)))
print('total validation cat images:', len(os.listdir(validation_cats_dir)))
print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
print('total test cat images:', len(os.listdir(test_cats_dir)))
print('total test dog images:', len(os.listdir(test_dogs_dir)))


from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.models import Sequential
from keras.layers import Dense, Input
import numpy as np 

#from tensorflow.keras.callbacks import TensorBoard
#from keras.layers.advanced_activations import LeakyReLU, PReLU

'''
# cropping images

import cv2

layers = [
     imageInputLayer([64 64 3],'Name','image')
     crop2dLayer('centercrop','Name','crop')
     ]
lgraph = layerGraph(layers)
lgraph = connectLayers(lgraph,'image','crop/ref')  

'''

# Initialising the CNN


'''
He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on 
imagenet classification." Proceedings of the IEEE international conference on computer 
vision. 2015.

classifier = Sequential()
act = keras.layers.advanced_activations.PReLU(init='zero', weights=None)
classifier.add(Dense(128, input_dim=14, init='uniform'))
# classifier.add(act)
'''
# another try to use PReLUs but uses Dropout which is not allowed

'''
classifier = Sequential()
act = keras.layers.advanced_activations.PReLU(init='zero', weights=None)
classifier.add(Dense(64, input_dim=14, init='uniform'))
classifier.add(Activation(act))
classifier.add(Dropout(0.15))
classifier.add(Dense(64, init='uniform'))
classifier.add(Activation('softplus'))
classifier.add(Dropout(0.15))
classifier.add(Dense(2, init='uniform'))
classifier.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
classifier.compile(loss='binary_crossentropy', optimizer=sgd)
classifier.fit(X_train, y_train, nb_epoch=20, batch_size=16, show_accuracy=True, validation_split=0.2, verbose = 2)
'''
from keras import layers


classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), activation='relu',
                        input_shape=(img_width, img_height, 3)))

# pooling
classifier.add(MaxPooling2D((2, 2)))

# first convolutional layer
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D((2, 2), strides=2))

# second layer
classifier.add(Conv2D(128, (3, 3), activation='relu'))
classifier.add(MaxPooling2D((2, 2), strides=2))

# third layer
classifier.add(Conv2D(128, (3, 3), activation='relu'))
classifier.add(MaxPooling2D((2, 2)))

# flattening
classifier.add(layers.Flatten())

# fully connected cnn
classifier.add(Dense(512, activation='relu'))
classifier.add(Dense(1, activation='sigmoid'))


classifier.summary()

#adam = tensorflow.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

from keras import optimizers

classifier.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


from keras.preprocessing.image import ImageDataGenerator


# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
#train_datagen = ImageDataGenerator(rescale=1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(img_width, img_height),
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')


validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')


for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break


import tensorflow as tf
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# config = tf.ConfigProto( device_count = {'GPU':0} ) 
#sess = tf.Session(config=config) 
#keras.backend.set_session(sess)



history = classifier.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=20,
      validation_data=validation_generator,
      validation_steps=50)

import matplotlib.pyplot as plt
#from keras.preprocessing import image
#test_image = image.load_img('dataset/single_prediction/cat_or_dog.jpg', target_size = (img_width, img_height))
#test_image = image.img_to_array(test_image)
#test_image = np.expand_dims(test_image, axis = 0)
#result = classifier.predict(test_image )
#training_set.class_indices
#if result [0][0] == 1:
#    prediction = 'dog'
#else:
#    prediction = 'cat'

#%matplotlib inline

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()