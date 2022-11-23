import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
data,metadata= tfds.load("cats_vs_dogs", as_supervised=True,
with_info=True) # data downloaded thanks to Microsoft, it has an image, a filename and a label
training = [] 
import matplotlib.pyplot as pl
import cv2
for i, (image, label) in enumerate(data["train"]):
    image=cv2.resize(image.numpy(),(100,100))
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #we changwe it top gray and resize it to make it faster
    image.reshape(100,100,1)
    training.append([image,label])
    x = []
    y=[]
for  image, label in training:
    x.append(image)
    y.append(label)

x= np.array(x).astype(float) /255 #we normalize it

y = np.array(y)


CNNModel= tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation= "relu", input_shape=(100,100,1)), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation= "relu", input_shape=(100,100,1)), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation= "relu", input_shape=(100,100,1)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation= "relu"),
    tf.keras.layers.Dense(1, activation= "sigmoid")
])


CNNModel.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
