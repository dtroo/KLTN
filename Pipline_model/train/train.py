from keras.models import Sequential,Model
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D
import pandas as pd
import numpy as np
import seaborn as sns
from keras.models import Sequential,Model
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from keras import layers
from tensorflow import keras
from keras import metrics
from sklearn.model_selection import train_test_split
#import joblib
import argparse
#from keras import backend as K

def train_model(x_train ,y_train, x_valid, y_valid ):


    x_train_data = np.load(x_train)
    y_train_data = np.load(y_train)

    model = Sequential()
    model.add(layers.Input(shape=(28,28,1)))
    model.add(Conv2D(64, (5, 5), activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.BatchNormalization()) 
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(MaxPool2D((3,3)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(2, activation='softmax'))
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',metrics.Recall(),metrics.Precision()])
    
    model.fit(np.array(x_train)/128.0,np.array(y_train),epochs=2,batch_size=100,validation_data=(np.array(x_valid)/128.0, np.array(y_valid)))
    
    model.save('model.h5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_train')
    parser.add_argument('--y_train')
    parser.add_argument('--x_valid')
    parser.add_argument('--y_valid')
    args = parser.parse_args()
    train_model(args.x_train, args.y_train, args.x_valid, args.y_valid)
    