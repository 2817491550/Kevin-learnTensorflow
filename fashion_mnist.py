import tensorflow as tf     
import tensorflow.keras as keras 
import numpy as np          
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import sys 
import pandas as pd 


fashion_mnist=keras.datasets.fashion_mnist
(train_x,train_y),(test_X,test_y)=fashion_mnist.load_data()
train_x=train_x/255.0
test_X=test_X/255.0
model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
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
history=model.fit(train_x,train_y,epochs=10,callbacks=[callbacks])
model.evaluate(test_X,test_y)
plt.plot(pd.DataFrame(history.history))
plt.show()

