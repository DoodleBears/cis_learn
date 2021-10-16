1. 为使用 xlearn 来进行 fm 训练的 data
import xlearn as xl

X_train = pd.read_csv('train_data/X_train.csv', header=None)
X_valid = pd.read_csv('train_data/X_valid.csv', header=None)
y_train = pd.read_csv('train_data/y_train.csv', header=None)
y_valid = pd.read_csv('train_data/y_valid.csv', header=None)
X_train = np.array(X_train)
X_valid = np.array(X_valid)
y_train = np.array(y_train)
y_valid = np.array(y_valid)
# print(X_train)

xdm_train = xl.DMatrix(X_train, y_train)
xdm_test = xl.DMatrix(X_valid, y_valid)

