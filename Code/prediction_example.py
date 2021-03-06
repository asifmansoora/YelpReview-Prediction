
# written by Rajnish, Dileep, Asif

import time
start_time = time.time()

from sklearn import linear_model
from sklearn.metrics import auc_score
import numpy as np
from sklearn import svm
from sklearn.gaussian_process import GaussianProcess
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from nltk.collocations import* 
from FormatText import * #external function file
from ModelFunctions import *
from FormatFeatures import *
from Text_Analysis import *
from logger import *

#Preprocessing data

from sklearn.cross_validation import KFold


#################################################################################
# MODEL PARAMETERS
#################################################################################

bCrossValidation = False
kFold = 5
bRunLiveYelpTest = True

sTrainingFile = 'consolidated_yelp_training_data.csv'
sTestingFile = 'consolidated_yelp_testing_data.csv'
sValidatinFile = 'consolidated_yelp_validation_data.csv'

n_neighbors = 10

#Regressor_Name = 'svr'
Regressor_Name = 'nusvr'
#Regressor_Name = 'Gausian'
#Regressor_Name = 'Nearest_Neighbors_uniform'
#Regressor_Name = 'Nearest_Neighbors_distance'

from nltk.stem import WordNetLemmatizer
wordNetLemma = WordNetLemmatizer()

IgnoreColumn = 'total_reviews_written_by_user'
 
#################################################################################
# LOADING TRAINING DATA
#################################################################################
 
trainfile = open(sTrainingFile)
header = trainfile.next().rstrip().split(',')
 
y_train = []
X_train = []

reviewData = []
iCount = 0 # calculates the number of sample 

print "\t Time is: "+str(time.time()-start_time)
print "LOADING THE TRAINING DATA"
for line in trainfile:
    splitted = line.rstrip().split(',')

    #   Processing Review Data  #
    wordList =  clean_review(splitted[1])   #To clean all the nonAlphaNumeric character
    wordList = filterListofWords(wordList)  #to Filter words less than length 3
    wordList = removeStopwords(wordList)     #remove stop words
    
    
    #wordList =  getPOSList(clean_review(splitted[1]),Noun=True)
    splitted[1] = len(wordList) #adding the feature, as length of document
    #wordList =  word_stemming(wordList) #I think Below function word_Lemmantization will give good result than stemming
    wordList = word_Lemmantization(wordList,wordNetLemma)
    
    reviewData.append(wordList)
    

    #continue
    
    label = int(splitted[0])
    features = []
    for item in splitted[1:30]:
        if(item == 'NULL'):
            features.append(np.nan)
        else:
            features.append(float(item))

        
    y_train.append(label)
    X_train.append(features)
    if(iCount == 100):
        
        break
    iCount = iCount +1
    
trainfile.close()


print "\t Data loaded succesfull: # of rows: ",len(X_train)," # of Columns: ",len(X_train[0]), " Length of review data: ",len(reviewData)
print "\t Time is: "+str(time.time()-start_time)+"\n"

print "\t Converting the Training Text Data to Topic Model"
TopicModel_data = TransformFeatureDoc(reviewData,"Training")  #Transform into featues
reviewData = []  #RAM Clean
reviewFeature = [] #New variale intialization
reviewFeature = getDocumentFeatures(TopicModel_data)


print "\t Converted into review features in Topics, it has # of rows: ",len(reviewFeature)," # of Columns: ",len(reviewFeature[0]), 
print "\t Time is: "+str(time.time()-start_time)+"\n"

#After we have converted all of our feature into number we fill convert it into np format
y_train = np.array(y_train)
X_train = np.array(X_train)


#Concatenating the Review Feature with rest of features
if(len(y_train) != len(reviewFeature)):
    raise "Lenght of X_train and reviewFeature is not matched"
else:
    X_train = np.concatenate((X_train, reviewFeature), axis=1)

print "\t Time is: "+str(time.time()-start_time)+"\n"
# Delete the unrequired columns from the Dataset
try:
    iIndex = header.index(IgnoreColumn)-1
    X_train = np.delete(X_train,iIndex,1)
    print "\t Deleted the unrequired columns"
except:
    print "\t No column matched for removing"



#Handle NULL(nan) values
X_train[:,2] = fill_avg(column(X_train,2))
print "\t Missing Values: Added mean values in the missing data"
print "\t Time is: "+str(time.time()-start_time)+"\n"
 
###########################
# 
###########################
 

print "\t Transformed the features"
X_train = transform_features_standardize(X_train) 
print "\t Time is: "+str(time.time()-start_time)+"\n"
###############################################################################
### SELECT THE MODEL TO RUN
###############################################################################




model = SelectModel(Regressor_Name)
model.fit(X_train,y_train)
print "\t Fit the model: ",Regressor_Name
print "\t Time is: "+str(time.time()-start_time)+"\n"
###############################################################################
# READING VALIDATION DATA
#
#
###############################################################################
print "LOADING THE VALIDATION DATA"
validationfile = open(sValidatinFile)
headerValidation = validationfile.next().rstrip().split(',')
 
y_validation = []
X_validation = []
reviewData = []
 
for line in validationfile:
    splitted = line.rstrip().split(',')

    #   Processing Review Data  #
    wordList =  clean_review(splitted[1])   #To clean all the nonAlphaNumeric character
    wordList = filterListofWords(wordList)  #to Filter words less than length 3
    wordList = removeStopwords(wordList)     #remove stop words
    
    
    #wordList =  getPOSList(clean_review(splitted[1]),Noun=True)
    splitted[1] = len(wordList) #adding the feature, as length of document
    #wordList =  word_stemming(wordList) #I think Below function word_Lemmantization will give good result than stemming
    wordList = word_Lemmantization(wordList,wordNetLemma)
    
    reviewData.append(wordList)
  

  
    label = int(splitted[0])
    features = []
    for item in splitted[1:30]:
        if(item == 'NULL'):
            features.append(np.nan)
        else:
            features.append(float(item))



    y_validation.append(label)
    X_validation.append(features)
    
validationfile.close()
print "\t Data loaded succesfull: # of rows: ",len(X_validation)," # of Columns: ",len(X_validation[0]), " Length of review data: ",len(reviewData)
print "\t Time is: "+str(time.time()-start_time)+"\n"

print "\t Converting the Validation Text Data to Topic Model"
TopicModel_data = TransformFeatureDoc(reviewData,"Validation")  #Transform into featues
reviewData = []  #RAM Clean
reviewFeature = [] #RAM Clean
reviewFeature = getDocumentFeatures(TopicModel_data)


print "\t Converted into review features in Topics, it has # of rows: ",len(reviewFeature)," # of Columns: ",len(reviewFeature[0]), 
print "\t Time is: "+str(time.time()-start_time)+"\n"


#After we have converted all of our feature into number we fill convert it into np format
y_validation = np.array(y_validation)
X_validation = np.array(X_validation)


#Concatenating the Review Feature with rest of features
if(len(X_validation) != len(reviewFeature)):
    raise "Lenght of X_validation and reviewFeature is not matched"
else:
    X_validation = np.concatenate((X_validation, reviewFeature), axis=1)


# Delete the unrequired columns from the Dataset
try:
    iIndex = header.index(IgnoreColumn)-1
    X_validation = np.delete(X_validation,iIndex,1)
    print "\t Deleted the unrequired columns" 
except:
    print "\t No column matched for removing"


#Handle NULL(nan) values

X_validation[:,2] = fill_avg(column(X_validation,2))


print "\t Missing Values: Added mean values in the missing data"   
print "\t Time is: "+str(time.time()-start_time)+"\n"

print "\t Transformed the features"
X_validation = transform_features_standardize(X_validation) 
print "\t Time is: "+str(time.time()-start_time)+"\n"

# compute result on the validation data 
print "\n Predicting the model result: \n"
y_predict = model.predict(X_validation)
print "\t Time is: "+str(time.time()-start_time)+"\n"


###############################################################################
# CALCULATING ACCURACY OF THE MODEL
###############################################################################
    
RMSLE = calculate_RMSLE(y_validation, y_predict) 
time_tak = time.time()-start_time
 
print "Mean Square Error on Validation data is: ",mean_squared_error(y_validation,y_predict)
print "R square error is: ",r2_score(y_validation, y_predict)                           
print "Root Mean Square Log Error: ", RMSLE 

# logging the result - Asif
logg_result(Regressor_Name,iCount,RMSLE, time_tak)

#Cross-Validation: evaluating estimator performance
if(bCrossValidation):
    print "Cross Validation: RMSLE is: ",RunCrossValidation(X_train,y_train,Regressor_Name,kFold)," after ",kFold," iteration"
               
                
                
###############################################################################
# READING TEST DATA
###############################################################################
print "LOAD THE TEST DATA..."
testfile = open(sTestingFile)
#Getting test header
testheader = testfile.next().rstrip().split(',')

X_test = []
review_id =[]
iIndex = testheader.index('review')

for line in testfile:
    splitted = line.rstrip().split(',')
    
    #   Processing Review Data  #
    wordList =  clean_review(splitted[iIndex])   #To clean all the nonAlphaNumeric character
    wordList = filterListofWords(wordList)  #to Filter words less than length 3
    wordList = removeStopwords(wordList)     #remove stop words
    
    #wordList =  getPOSList(clean_review(splitted[1]),Noun=True)
    splitted[iIndex] = len(wordList) #adding the feature, as length of document
    #wordList =  word_stemming(wordList) #I think Below function word_Lemmantization will give good result than stemming
    wordList = word_Lemmantization(wordList,wordNetLemma)
    
    reviewData.append(wordList)
 
    features = []

    #Adding review Id for test file
    review_id.append(splitted[29])
    for item in splitted[0:29]:
        if(item == 'NULL'):
            features.append(np.nan)
        else:
            features.append(float(item))

    
    X_test.append(features)
 
testfile.close()

print "\t Data loaded succesfull: # of rows: ",len(X_test)," # of Columns: ",len(X_test[0]), " Length of review data: ",len(reviewData)
print "\t Time is: "+str(time.time()-start_time)+"\n"

print "\t Converting the Testing Text Data to Topic Model"
TopicModel_data = TransformFeatureDoc(reviewData,"test")  #Transform into featues
reviewData = []  #RAM Clean
reviewFeature = [] #RAM Clean
reviewFeature = getDocumentFeatures(TopicModel_data)

#Concatenating the Review Feature with rest of features
if(len(X_test) != len(reviewFeature)):
    raise "Lenght of X_test and reviewFeature is not matched"
else:
    X_test = np.concatenate((X_test, reviewFeature), axis=1)


print "\t Converted into review features in Topics, it has # of rows: ",len(reviewFeature)," # of Columns: ",len(reviewFeature[0]), 
print "\t Time is: "+str(time.time()-start_time)+"\n"


X_test = np.array(X_test)


# Delete the unrequired columns from the Dataset
try:
    iIndex = testheader.index(IgnoreColumn)
    X_test = np.delete(X_test,iIndex,1)
    print "\t Deleted the unrequired columns: ",IgnoreColumn,
except:
    print "\t No column matched for removing"



#Handle NULL(nan) values
X_test[:,2] = fill_avg(column(X_test,2))
print "\t Missing Values: Added mean values in the missing data"

 

###############################################################################
# TRANSFORMING THE FEATURES 
###############################################################################
print "\t Transformed the features"
X_test = transform_features_standardize(X_test) 
print "\t Time is: "+str(time.time()-start_time)+"\n"

###############################################################################
# PREDICT FROM THE MODEL
###############################################################################
                
p_test = model.predict(X_test)

##############################################################################
# WRITING SUBMISSION FILE
###############################################################################
prediction = open("prediction.csv","w")
prediction.write(str(("id,votes"+"\n")))
for i,j in zip(review_id,p_test):
    prediction.write(str((i+","+str(j)+"\n")))
prediction.close()


