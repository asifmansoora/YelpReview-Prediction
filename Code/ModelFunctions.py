from sklearn.gaussian_process import GaussianProcess
from sklearn import neighbors
from sklearn import svm
import numpy as np
from sklearn import preprocessing
import math

####################################################################################################
## MODEL FUNCTIONS
####################################################################################################

def calculate_RMSLE(y_test,y_predict):
    error = [ math.pow((math.log(y_predict+1)-math.log(y_test+1)),2) for y_predict,y_test in zip(y_predict,y_test)]
    err_len = len(error)
    error_sum = sum(error)

    error = math.sqrt(error_sum/err_len)
    return error   


def SelectModel(regressor):

    if(regressor == 'svr'):
        model = svm.SVR()
    elif (regressor == 'nusvr'):
        model = svm.NuSVR()
    elif (regressor == 'Gausian'):
        model = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1e-1,random_start=100)
    elif (regressor == 'Nearest_Neighbors_uniform'):
        model = neighbors.KNeighborsRegressor(n_neighbors, weights='uniform')
    elif (regressor == 'Nearest_Neighbors_distance'):
        model = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
    
    return model

def transform_features_log(x):
    return np.log(1+x)

# To Convert into Scaled data that has zero mean and unit variance:
def transform_features_standardize(x):
    return preprocessing.scale(x)


def RunCrossValidation(X, Y, modelName,k):

    listOfError = []
    kf = KFold(len(Y), k, shuffle =True)
    for train_index, test_index in kf:
        
        #print "TRAIN:", train_index, "TEST:", test_index
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        
        model = SelectModel(regressor)
        model.fit(X_train,y_train)
        y_predict = model.predict(X_test)
        error = calculate_RMSLE(y_test,y_predict)
        listOfError.append(error)

    listOfError = np.array(listOfError)
    return np.mean(listOfError)
