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


def retrain_model(x_train, y_train, x_valid, y_valid, model_path):
    model = load_model(model_path)
    model.fit(np.array(X_train)/128.0,np.array(y_train),epochs=2,batch_size=100,validation_data=(np.array(X_valid)/128.0, np.array(y_valid)))
    model.save('model.h5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_train')
    parser.add_argument('--y_train')
    parser.add_argument('--x_valid')
    parser.add_argument('--y_valid')
    parser.add_argument('--model')
    args = parser.parse_args()
    train_model(args.x_train, args.y_train, args.x_valid, args.y_valid, args.model)
    