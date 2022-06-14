import argparse
import numpy as np
from sklearn.metrics import mean_squared_error
from keras.models import load_model

def test_model(x_test, y_test, model_path):
    x_test_data = np.load(x_test)
    y_test_data = np.load(y_test)

    model = load_model(model_path)
    y_pred = model.predict(np.array(x_test_data))

    err = mean_squared_error(y_test_data, y_pred)
    
    with open('output.txt', 'a') as f:
        f.write(str(err))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_test')
    parser.add_argument('--y_test')
    parser.add_argument('--model')
    args = parser.parse_args()
    test_model(args.x_test, args.y_test, args.model)