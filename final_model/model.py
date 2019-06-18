import cv2
import tensorflow as tf

def prepare(filepath):
	img_array = cv2.imread(filepath)
    	img = cv2.resize(img_array, (50,100)) 
    	gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    	norm_image=cv2.normalize(gray_image,gray_image, 0, 255, cv2.NORM_MINMAX)
    	smooth_image = cv2.GaussianBlur(norm_image,(5,5),0)
    

model = tf.keras.models.load_model("64x2-CNN.model")
prediction = model.predict([prepare(img)])
