import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

data = keras.datasets.mnist

data

df = data.load_data()

(X_train, y_train),(X_test, y_test) = df
X_train.shape
X_test.shape
y_test.shape
y_train.shape
X_train[0].ndim
y_test.shape
y_test.ndim
X_test.ndim

X_train_flat = X_train.reshape(-1, 28*28)
X_test_flat = X_test.reshape(-1, 28*28)
X_train_flat[0].ndim

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
                    #First hidden layer
                    Dense(units=10,
                          input_shape=(784,),
                          activation='sigmoid')
                    
                    
                    
                    
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train_flat,y_train, epochs=10)
y_pred = model.predict(X_test_flat)
y_pred
y_test

model.evaluate(X_test_flat, y_test)

#Aswanth Anoop
#aswanthanoop42@gmail.com