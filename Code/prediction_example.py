#!/usr/bin/python
# -*- coding: utf8 -*-
 
#
# written by Ferenc Huszár, PeerIndex
 
from sklearn import linear_model
from sklearn.metrics import auc_score
import numpy as np
from sklearn import svm
from sklearn.gaussian_process import GaussianProcess
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#Preprocessing data
from sklearn import preprocessing
from sklearn.cross_validation import KFold


#################################################################################
# MODEL PARAMETERS
#################################################################################

bCrossValidation = False



 
#################################################################################
# LOADING TRAINING DATA
#################################################################################
 
trainfile = open('consolidated_yelp_train_data_with_categories.csv')
header = trainfile.next().rstrip().split(',')
 
y_train = []
X_train = []

 
for line in trainfile:
    splitted = line.rstrip().split(',')
    
    label = int(splitted[0])
    features = [float(item) for item in splitted[1:]
    #B_features = [float(item) for item in splitted[12:]]
    y_train.append(label)
    X_train.append(features)
    
trainfile.close()




#After we have converted all of our feature into number we fill convert it into np format
y_train = np.array(y_train)
X_train = np.array(X_train)

 
###########################
# EXAMPLE BASELINE SOLUTION USING SCIKIT-LEARN
#
# using scikit-learn LogisticRegression module without fitting intercept
# to make it more interesting instead of using the raw features we transform them logarithmically
# the input to the classifier will be the difference between transformed features of A and B
# the method roughly follows this procedure, except that we already start with pairwise data
# http://fseoane.net/blog/2012/learning-to-rank-with-scikit-learn-the-pairwise-transform/
###########################
 
def transform_features_log(x):
    return np.log(1+x)

# To Convert into Scaled data that has zero mean and unit variance:
def transform_features_standardize(x):
    return preprocessing.scale(X)



X_train = transform_features_log(X_train) 


###############################################################################
### SELECT THE MODEL TO RUN
###############################################################################

#run SVM
#clf = svm.NuSVR()	
#model = svm.SVR()
#model = svm.SVC(kernel='linear')
#model = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1e-1,random_start=100)

"""n_neighbors = 14
knn = neighbors.KNeighborsRegressor(n_neighbors, weights='uniform')
knn = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
"""
#model.fit(X_train,y_train)




model = linear_model.LogisticRegression(fit_intercept=True)
model.fit(X_train,y_train)



###############################################################################
# READING VALIDATION DATA
###############################################################################

validationfile = open('consolidated_yelp_validation_data.csv')
headerValidation = validationfile.next().rstrip().split(',')
 
y_validation = []
X_validation = []

 
for line in validationfile:
    splitted = line.rstrip().split(',')
    
    label = int(splitted[0])
    features = [float(item) for item in splitted[1:]
    #B_features = [float(item) for item in splitted[12:]]
    y_validation.append(label)
    X_validation.append(features)
    
validationfile.close()




#After we have converted all of our feature into number we fill convert it into np format
y_validation = np.array(y_validation)
X_validation = np.array(X_validation)


# compute result on the validation data 
y_predict = model.predict(X_validation)




###############################################################################
# CALCULATING ACCURACY OF THE MODEL
###############################################################################
 
print "Mean Square Error on Validation data is: ",mean_squared_error(y_validation,y_predict)
print "R square error is: ",r2_score(y_true, y_pred)                           


#Cross-Validation: evaluating estimator performance
if(bCrossValidation):
                


                


###############################################################################
# READING TEST DATA
###############################################################################
 
testfile = open('test.csv')
#ignore the test header
testfile.next()
 
X_test = []

for line in testfile:
    splitted = line.rstrip().split(',')
    features = [float(item) for item in splitted]
    X_test.append(features)
    
testfile.close()
 
X_test = np.array(X_test)

###############################################################################
# TRANSFORMING THE FEATURES 
###############################################################################

X_test = transform_features_log(X_test)



###############################################################################
# PREDICT FROM THE MODEL
###############################################################################
                
p_test = model.predict(X_test)
                
###############################################################################
# WRITING SUBMISSION FILE
###############################################################################
predfile = open('predictions.csv','w+')
 
print >>predfile,','.join(["review_id","class"])
for line in np.concatenate((X_test,p_test),axis=1):
    print >>predfile, ','.join([str(item) for item in line])
 
predfile.close()
