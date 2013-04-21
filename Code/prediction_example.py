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
from nltk.stem import WordNetLemmatizer
from nltk.collocations import* 
from FormatText import * #external function file
from ModelFunctions import *

wordNetStemmer = WordNetLemmatizer()

#Preprocessing data

from sklearn.cross_validation import KFold


#################################################################################
# MODEL PARAMETERS
#################################################################################

bCrossValidation = False
kFold = 5
bRunLiveYelpTest = False

sTrainingFile = 'consolidated_yelp_training_data.csv'
sTestingFile = 'consolidated_yelp_testing_data.csv'
sValidatinFile = 'consolidated_yelp_validation_data.csv'

n_neighbors = 10

Regressor_Name = 'svr'
#Regressor_Name = 'nusvr'
#Regressor_Name = 'Gausian'
#Regressor_Name = 'Nearest_Neighbors_uniform'
#Regressor_Name = 'Nearest_Neighbors_distance'
 
#################################################################################
# LOADING TRAINING DATA
#################################################################################
 
trainfile = open(sTrainingFile)
header = trainfile.next().rstrip().split(',')
 
y_train = []
X_train = []

reviewData = []
iCount = 0

for line in trainfile:
    splitted = line.rstrip().split(',')
    
    wordList =  clean_review(splitted[1])
    #print wordList
    
    #wordList =  getPOSList(clean_review(splitted[1]),Noun=True)
    splitted[1] = len(wordList)
    
    # print word_stemming(wordList,wordNetStemmer)
    """
    if(iCount == 10):
        break
    iCount = iCount +1
    continue
    """
    label = int(splitted[0])
    features = []
    for item in splitted[1:30]:
        if(item == 'NULL'):
            features.append(item)
        else:
            features.append(float(item))

        
    #features = [if(item != 'NULL') float(item) for item in splitted[1:30]]
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
 


#X_train = transform_features_standardize(X_train) 


###############################################################################
### SELECT THE MODEL TO RUN
###############################################################################




model = SelectModel(Regressor_Name)
model.fit(X_train,y_train)


###############################################################################
# READING VALIDATION DATA
###############################################################################

validationfile = open(sValidatinFile)
headerValidation = validationfile.next().rstrip().split(',')
 
y_validation = []
X_validation = []

 
for line in validationfile:
    splitted = line.rstrip().split(',')
  
    wordList = clean_review(splitted[1])
    splitted[1] = len(wordList)
  
    label = int(splitted[0])
    features = []
    for item in splitted[1:30]:
        if(item == 'NULL'):
            features.append(item)
        else:
            features.append(float(item))
    #features = [float(item) for item in splitted[1:]]
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
print "R square error is: ",r2_score(y_validation, y_predict)                           
print "Root Mean Square Log Error: ", calculate_RMSLE(y_validation, y_predict) 

1/0
#Cross-Validation: evaluating estimator performance
if(bCrossValidation):
    print "Cross Validation: RMSLE is: ",RunCrossValidation(X_train,y_train,Regressor_Name,kFold)," after ",kFold," iteration"
               
                
                
###############################################################################
# READING TEST DATA
###############################################################################
 
testfile = open(sTestingFile)
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

X_test = transform_features_standardize(X_test)



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
