from ImagePreprocessing import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

N_Classes = 43
Resized_Image = (32, 32)

dataset = preprocessing(N_Classes, Resized_Image)
print(dataset.X.shape)
print(dataset.y.shape)

idx_train, idx_test = train_test_split(range(dataset.X.shape[0]), test_size = 0.25, random_state = 101)

X_train = dataset.X[idx_train, :, :, :]
X_test = dataset.X[idx_test, :, :, :]
y_train = dataset.y[idx_train, :]
y_test = dataset.y[idx_test, :]

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

