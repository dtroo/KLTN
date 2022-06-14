from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import requests

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

def reparing_data():
  #### check model exists
  URL_1 = "https://raw.githubusercontent.com/dtroo/KLTN/main/New_requests/goodrequests.txt"
  response_1 = requests.get(URL_1)
  URL_2 = "https://raw.githubusercontent.com/dtroo/KLTN/main/New_requests/badrequests.txt"
  response_2 = requests.get(URL_2)
  if response_1.status_code == 404 and response_2.status_code == 404: 
    return False
  open("goodrequests.txt", "wb").write(response_1.content)
  open("badrequests.txt", "wb").write(response_2.content)
  return True

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
  X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=0.2, random_state=42)
  X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)
  np.save('x_train.npy', X_train)
  np.save('x_test.npy', X_test)
  np.save('x_valid.npy', X_valid)
  np.save('y_train.npy', y_train)
  np.save('y_test.npy', y_test)
  np.save('y_valid.npy', y_valid)

def check_model_is_exists():
  ## url model 
  URL = "https://github.com/dtroo/KLTN/raw/main/Model/model.h5"
  r = requests.get(URL)
  if r.status_code == 200:
    return True
  return False

if __name__ == '__main__':
     
  # check to do retraining or crate new model
  if(check_model_is_exists() == True):
    print('Retraining data to model....')_
    if(reparing_data() ==  False):
      print("new data not found!.....")
    else:
      print('retraining model......')
      _preprocess_data()
      np.save('retrain.npy','1')
  # create new model
  else:
    print('Preprocessing data to new model...')
    _preprocess_data()
