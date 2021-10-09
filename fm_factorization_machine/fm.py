# fm training for anime

import xlearn as xl
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# 读取 csv
anime_train_df = pd.read_csv("data/anime_genres_splitted.csv" , header=None)
col_num = len(anime_train_df.columns.tolist())

# 采样 sampling
x = anime_train_df
kf = KFold(n_splits=10,shuffle=True)
shuffle_index = 0
for train_index, test_index in kf.split(x):
  shuffle_index+=1
  print(train_index,test_index)

  # 拆分 test 和 train
  # drop 掉 test 的部分
  train_sample = anime_train_df.drop(train_index)
  X_train = train_sample[train_sample.columns[1:]]
  y_train = train_sample[0]

  # drop 掉 train 的部分
  test_sample = anime_train_df.drop(test_index)

  X_test = test_sample[test_sample.columns[1:]]
  y_test = test_sample[0]
  # test_sample

  # 转化为 xlearn 所要求的格式
  xdm_train = xl.DMatrix(X_train, y_train)
  xdm_test = xl.DMatrix(X_test, y_test)

  # 开始 train
  # Training task
  fm_model = xl.create_fm()  # Use factorization machine
  # we use the same API for train from file
  # that is, you can also pass xl.DMatrix for this API now
  fm_model.setTrain(xdm_train)    # Training data
  fm_model.setValidate(xdm_test)  # Validation data

  # param:
  #  0. regression task
  #  1. learning rate: 0.2
  #  2. regular lambda: 0.002
  #  3. evaluation metric: acc
  param = {'task':'binary', 'lr':0.1, 
          'lambda':0.002, 'metric':'acc', 'k':2}

  # Start to train
  # The trained model will be stored in model.out
  fm_model.fit(param, './model_dm.out')

  # Prediction task
  # we use the same API for test from file
  # that is, you can also pass xl.DMatrix for this API now
  fm_model.setTest(xdm_test)  # Test data
  # fm_model.setSign()  # Convert output to 0, 1
  # fm_model.setSigmoid()  # Convert output to 0-1

  # Start to predict
  # The output result will be stored in output.txt
  # if no result out path setted, we return res as numpy.ndarray
  path = "./output" + str(shuffle_index) + ".txt"
  fm_model.predict("./model_dm.out", path)
  # res = fm_model.predict("./model_dm.out", "./output.txt")

  # print(res)

