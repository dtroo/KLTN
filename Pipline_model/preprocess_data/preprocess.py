from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def generate_dataset(df):
  tensorlist = []
  labelist = []

  # Generating our dataset sample by sample
  for row in df.values:

    query = row[0]
    label = row[1]

    # Encode characters into their UNICODE value
    url_part = [ord(x) if (ord(x) < 129) else 0 for x in query[:784]]

    # Pad with zeroes
    url_part += [0] * (784 - len(url_part))
    
    maxim = max(url_part)
    maxabs = 0
    if maxim > maxabs:
      maxabs = maxim
    x = np.array(url_part).reshape(28,28)
    
    # label y
    if label == 1:
        y = np.array([0, 1], dtype=np.int8)
    else :
        y = np.array([1, 0], dtype=np.int8)
    tensorlist.append(x)
    labelist.append(y)

  return tensorlist,labelist

def _preprocess_data():
     # load data
     f = open('./badrequests.txt',mode='r')
     badqueries = f.readlines()
     f = open('./goodrequests.txt',mode='r')
     goodqueries = f.readlines()
     f.close()

     # preprocess
     badqueriespd = pd.DataFrame({'query': badqueries,'label': [1 for x in badqueries]})
     goodqueries = pd.DataFrame({'query': goodqueries,'label': [0 for x in goodqueries]})
     allqueries = pd.concat([badqueriespd,goodqueries])

     # create dataset and save it to file
     X,y = generate_dataset(allqueries)
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     np.save('x_train.npy', X_train)
     np.save('x_test.npy', X_test)
     np.save('y_train.npy', y_train)
     np.save('y_test.npy', y_test)
     
if __name__ == '__main__':
     print('Preprocessing data...')
     _preprocess_data()
