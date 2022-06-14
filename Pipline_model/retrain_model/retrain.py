from keras.models import Sequential,Model,load_model
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
import argparse
from keras import backend as K
import tensorflow as tf
import requests


def retrain_model(x_train, y_train, x_valid, y_valid, model):
    x_train_data = np.load(x_train)
    y_train_data = np.load(y_train)
    x_valid_data = np.load(x_valid)
    y_valid_data = np.load(y_valid)
    model.fit(np.array(x_train_data),np.array(y_train_data),epochs=2,batch_size=100,validation_data=(np.array(x_valid_data), np.array(y_valid_data)))
    model.save('model.h5')

def get_model():
    URL = "https://github.com/dtroo/KLTN/raw/main/Model/model.h5"
    r = requests.get(URL)
    if (r.status_code == 200):
        open("model.h5", "wb").write(r.content)
        
    else: 
        print('Model not found!...')

if __name__ == '__main__':
    get_model()
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_train')
    parser.add_argument('--y_train')
    parser.add_argument('--x_valid')
    parser.add_argument('--y_valid')
    model = load_model('model.h5')
    args = parser.parse_args()
    retrain_model(args.x_train, args.y_train, args.x_valid, args.y_valid, model)
    