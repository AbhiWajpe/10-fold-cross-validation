import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

trng = np.loadtxt("C:\\Users\\wajpe\\OneDrive\\Desktop\\Lectures\\AESRP 597\\HW SET\\HW1\\flight_data_train.csv", delimiter=',',
                encoding='utf8')

test = np.loadtxt("C:\\Users\\wajpe\\OneDrive\\Desktop\\Lectures\\AESRP 597\\HW SET\\HW1\\flight_data_test.csv", delimiter=',',
                encoding='utf8')

dataset = np.concatenate((trng,test),axis=0)

X = dataset[:,0:6]
Y = dataset[:,6]
N = len(dataset)
k = 10
kf = KFold(n_splits=k, random_state=None)

training_error_all = []
testing_error_all = []

def range_of_lambda(l):
    l = np.exp(l)
    w_train = np.linalg.inv(phi_train.T @ phi_train + l * np.eye(37)) @ phi_train.T @ y_train
    y_train_pred = np.dot(phi_train,w_train)
    error = np.sqrt(np.mean((y_train_pred-y_train)**2))
    return error, w_train

for train_index , test_index in kf.split(X):
    X_train , X_test = X[train_index,:], X[test_index,:]
    y_train , y_test = Y[train_index] , Y[test_index]
    phi_train = np.zeros([len(X_train),36])
    for i in range(1,7):
        phi_train[:,6*(i-1):(6*i)]  = X_train[:,0:6]**i
    phi_train_max = np.max(phi_train, axis=0, keepdims=True)
    phi_train_min = np.min(phi_train, axis=0, keepdims=True)
    phi_train = (phi_train - phi_train_min)/(phi_train_max - phi_train_min)
    ones_array = np.ones([len(phi_train),1])
    phi_train = np.c_[ones_array,phi_train]
    phi_test = np.zeros([len(X_test),36])
    for i in range(1,7):
        phi_test[:,6*(i-1):(6*i)]  = X_test[:,0:6]**i
    phi_test_max = np.max(phi_test, axis=0, keepdims=True)
    phi_test_min = np.min(phi_test, axis=0, keepdims=True)
    phi_test = (phi_test - phi_train_min)/(phi_train_max - phi_train_min)
    ones_array = np.ones([len(phi_test),1])
    phi_test = np.c_[ones_array,phi_test]
    
    weights = [0]*41
    y_test_pred = [0]*41
    training_error = np.zeros(41)
    testing_error = np.zeros(41)
    lambdas = list(range(-30, 11))
    for i, l in enumerate(lambdas):
        training_error[i], weights[i] = range_of_lambda(l)
        y_test_pred[i] = np.dot(phi_test,weights[i])
        testing_error[i] = np.sqrt(np.mean((y_test_pred[i]-y_test)**2))
    
    testing_error_all.append(testing_error)
    training_error_all.append(training_error)
    


testing_error_all = np.array(testing_error_all)
training_error_all = np.array(training_error_all)
RMSE_test_avg = np.mean(testing_error_all,axis=0)
RMSE_training_avg = np.mean(training_error_all,axis=0)

plt.plot(list(range(-30,11)),RMSE_training_avg,marker='.',markersize=7.5)
plt.plot(list(range(-30,11)),RMSE_test_avg,marker='.',markersize=7.5)
plt.xlabel('$ln\lambda$')
plt.ylabel('RMS Error')
plt.legend(['Training Error','Test Error'])
plt.title('Training error and Test error as RMSE against $ln\lambda$')
plt.show()


