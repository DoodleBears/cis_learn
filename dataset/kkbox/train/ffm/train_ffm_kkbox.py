import xlearn as xl

# 注意后面的_80 后缀和 _20 后缀代表的是 load data时 train占总数据的 百分比(%)
train_file_path = "data/ffm_train_data_1.txt"
test_file_path = "data/ffm_test_data_20.txt"

# 当 k = 4 80% 的 train 结果为: Early-stopping at epoch 78, best Accuarcy: 0.723808
# 当 k = 4 20% 的 train 结果为: Early-stopping at epoch 51, best Accuarcy: 0.699185
# 当 k = 4 1% 的 train 结果为: Early-stopping at epoch 16, best Accuarcy: 0.631143

ffm_model = xl.create_ffm() # Use field-aware factorization machine
ffm_model.setTrain(train_file_path)  # Training data
ffm_model.setValidate(test_file_path)  # Validation data

# param:
#  0. binary classification
#  1. learning rate: 0.2
#  2. regular lambda: 0.002
#  3. evaluation metric: accuracy

param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric':'acc', 'k':4, 'epoch': 100}
# ffm_model.setOnDisk()   #  Set xlearn to use on-disk training.
# Start to train
# The trained model will be stored in model.out
ffm_model.fit(param, './ffm_model.out')
# ffm_model.cv(param)    # Perform cross-validation.

# Prediction task
ffm_model.setTest(test_file_path)  # Validation data
# ffm_model.setTest("./small_test.txt")  # Test data
# ffm_model.setSigmoid()  # Convert output to 0-1
ffm_model.setSign()  # Convert output to 0-1

# Start to predict
# The output result will be stored in output.txt
ffm_model.predict("./ffm_model.out", "./ffm_output.txt")