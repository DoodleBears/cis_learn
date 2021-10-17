import xlearn as xl
import sys

test_file_path = "data/fm_test_data_100.txt"

index = int(sys.argv[1])

print("data/fm_train_data_{}.txt".format(index))
train_file_path = "data/fm_train_data_{}.txt".format(index)

# 开始 train
fm_model = xl.create_fm()  # Use factorization machine
# fm_model.setNoBin()
fm_model.setTrain(train_file_path)  # Validation data
fm_model.setValidate(test_file_path)  # Validation data

# fm_model.disableEarlyStop()
# param
param = {'task':'binary', 'lr':1.2, 'k': 4,
         'lambda':0.002, 'metric':'acc', 'epoch': 100 , 'stop_window': 3}

# Start to train
# The trained model will be stored in model.out
fm_model.fit(param, './fm_model_{}.out'.format(index))

# fm_model.cv(param)    # Perform cross-validation.
# Prediction task
fm_model.setTest(test_file_path)  # Test data
fm_model.setSign()  # Convert output to 0-1

# Start to predict
res = fm_model.predict('./fm_model_{}.out'.format(index), "./fm_output_{}.txt".format(index))