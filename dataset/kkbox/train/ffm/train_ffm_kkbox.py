# 运行方式(How to run): python train_ffm_kkbox.py <suffix-of-data-name>
# 这边使用 100% 的 train set 去 train 则: python train_ffm_kkbox.py 100
# 这边使用 20% 的 train set 去 train 则: python train_ffm_kkbox.py 20

import xlearn as xl
import sys

# 注意后面的_80 后缀和 _20 后缀代表的是 load data时 train占总数据的 百分比(%)
test_file_path = "data/ffm_test_data_20.txt"

# 读入输入的变量
index = int(sys.argv[1])

print("data/ffm_train_data_{}.txt".format(index))
train_file_path = "data/ffm_train_data_{}.txt".format(index)

ffm_model = xl.create_ffm() # Use field-aware factorization machine
ffm_model.setTrain(train_file_path)  # Training data
ffm_model.setValidate(test_file_path)  # Validation data

param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric':'auc', 'k':4, 'epoch': 100}
# ffm_model.setOnDisk()   #  Set xlearn to use on-disk training.
# Start to train
ffm_model.fit(param, './ffm_model_{}.out'.format(index))
# ffm_model.cv(param)    # Perform cross-validation.

# Prediction task
ffm_model.setTest(test_file_path)  # Validation data
ffm_model.setSign()  # Convert output to 0-1

# Start to predict
# The output result will be stored in output.txt
ffm_model.predict("./ffm_model_{}.out".format(index), "./ffm_output_{}.txt".format(index))