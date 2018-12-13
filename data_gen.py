import keras
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array, load_img
import os
import cv2

train_datagen=ImageDataGenerator(rotation_range=10,
                           width_shift_range=4,
                           height_shift_range=4,
                           rescale=1/255.0,
                           horizontal_flip=False,
                           vertical_flip=False,
                           fill_mode='nearest')


image_folder = "E:/DL/signature_classifier/trainingSet/OfflineSignatures/Dutch/TrainingSet/OfflineForgeries"
files=os.listdir(image_folder)
files=list(map(lambda x: os.path.join(image_folder,x),files))
a=(len(files))
ls_train=[]
for i in range(a):
    im = (files[i])
    s=100+int(im[-9:-7])
    img=cv2.imread(im) 
    img = img.reshape((1,) + img.shape)
    i = 0
    for batch in train_datagen.flow(img, batch_size=1,save_to_dir='E:/DL/signature_classifier/g', save_prefix=s, save_format='png'):
        #print(batch.shape)
        i += 1
        if i >5 : ## making 10
            break  # otherwise the generator would loop indefinitely

