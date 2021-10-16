import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xlearn as xl

merge_train_df = pd.read_csv('../../data_cleaned/merge_train_songs_members.csv')

y_full = pd.DataFrame(merge_train_df.target)

X_full = merge_train_df.drop(columns=['target'], axis=1)
X_full = X_full.drop(columns=['genre_ids'], axis=1)
# X_full = X_full.drop(columns=['msno', 'song_id'], axis=1)
# X_full = X_full.drop(columns=['song_length', 'artist_name', 'composer', 'lyricist'])
X_full = X_full.drop(columns=['registration_init_time', 'expiration_date'])
X_full = X_full.drop(columns=['registered_via'])

X_train, X_valid, y_train, y_valid = train_test_split(X_full, y_full, train_size=0.2, test_size= 0.2, shuffle=False, random_state=0)

# X_train = pd.read_csv('data/X_train.csv', header=None)
# X_valid = pd.read_csv('data/X_valid.csv', header=None)
# y_train = pd.read_csv('data/y_train.csv', header=None)
# y_valid = pd.read_csv('data/y_valid.csv', header=None)
# X_train = np.array(X_train)
# X_valid = np.array(X_valid)
# y_train = np.array(y_train)
# y_valid = np.array(y_valid)
# print(X_train)

xdm_train = xl.DMatrix(X_train, y_train)
xdm_test = xl.DMatrix(X_valid, y_valid)

# 开始 train
# Training task
fm_model = xl.create_fm()  # Use factorization machine
# fm_model.setTrain('merge_train.txt')    # Training data
# fm_model.setNoBin()
# fm_model.setValidate('merge_test.txt')  # Validation data
fm_model.setTrain(xdm_train)    # Training data
fm_model.setValidate(xdm_test)  # Validation data
# fm_model.disableEarlyStop()
# param
param = {'task':'binary', 'lr':0.2, 'k': 3,
         'lambda':0.002, 'metric':'acc', 'epoch': 200, 'opt':'sgd'}

# Start to train
# The trained model will be stored in model.out
fm_model.fit(param, './model_dm.out')

# fm_model.cv(param)    # Perform cross-validation.
# Prediction task
fm_model.setTest(xdm_test)  # Test data
fm_model.setSign()  # Convert output to 0-1

# Start to predict
res = fm_model.predict("./model_dm.out", "./output.txt")