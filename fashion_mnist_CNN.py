import tensorflow as tf     
import tensorflow.keras as keras 
import numpy as np          
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import sys 
import pandas as pd 
from tensorflow.keras import models 



fashion_mnist=keras.datasets.fashion_mnist
(train_x,train_y),(test_X,test_y)=fashion_mnist.load_data()
train_x=train_x.reshape(60000,28,28,1)
train_x=train_x/255.0
test_X=test_X.reshape(10000,28,28,1)
test_X=test_X/255.0
model=keras.Sequential([
    keras.layers.Conv2D(64,(3,3),activation='selu',input_shape=(28,28,1)),
    keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='selu'),
    tf.keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(128,activation=tf.nn.selu),
    keras.layers.Dense(10,activation=tf.nn.softmax)
])
class FitCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('loss')<0.3):
            print("\nloss is now so cancelling training!")
            self.model.stop_training=True
callbacks=FitCallback()
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy')
print(model.summary())
history=model.fit(train_x,train_y,epochs=5,callbacks=[callbacks])
#history=model.fit(train_x,train_y,epochs=10,callbacks=[callbacks])
#model.evaluate(test_X,test_y)
test_loss=model.evaluate(test_X,test_y)




