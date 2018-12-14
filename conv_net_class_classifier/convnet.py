import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import model_from_json
import cv2
import numpy as np
import os
import pandas as pd 
import matplotlib.pyplot as plt 
from random import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm

ls_train = []
image_folder='E:/DL/signature_classifier/g'
def create_training_data():
    for img in tqdm(os.listdir(image_folder)):
        path = os.path.join(image_folder,img)
        class_num = int(path[29:32])
        img_array = cv2.imread(path)
        img = cv2.resize(img_array, (50,100)) 
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        norm_image=cv2.normalize(gray_image,gray_image, 0, 255, cv2.NORM_MINMAX)
        smooth_image = cv2.GaussianBlur(norm_image,(5,5),0)
        ls_train.append([smooth_image,class_num])

create_training_data()
shuffle(ls_train)
X = []
y = []

for feature,label in ls_train:
    X.append(feature)
    y.append(label)

X = np.array(X).reshape(-1, 50, 100, 1)
X = X/255.0

num_classes=89
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same',input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.7))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.7))

model.add(Flatten())

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.7))
model.add(Dense(num_classes))
model.add(Activation(tf.nn.softmax))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

batch_size=32
epochs=150
model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_test, y_test),
              shuffle=True)

scores = model.evaluate(X_test, y_test, verbose=1)

model.save('64x2-CNN.model')
