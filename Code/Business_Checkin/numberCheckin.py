import re
import numpy as np

# CSV files used, to count the number of people check-in for a business
sTraining_CheckinFile = 'yelp_academic_dataset_checkin.csv'
sTesting_CheckinFile = 'yelp_test_set_checkin.csv'

train_Checkingfile = open(sTraining_CheckinFile)
test_Checkingfile = open(sTesting_CheckinFile)

header = train_Checkingfile.next().rstrip().split(',')
header_test = test_Checkingfile.next().rstrip().split(',')
print header
print header_test

number_of_checkin_train = []
businessid_train = []

number_of_checkin_test = []
businessid_test = []

# Splitting the training file to generate number checked alone and summing it up
for line_train in train_Checkingfile:
    splitted_train = line_train.rstrip().split(',') 
    businessid_train.append(splitted_train[1])
    str_rev_train = re.sub(r'[\{\}]',"",splitted_train[0])
    split_train = str_rev_train.split(";")
    result_train = []
    for num in split_train:
        values_train = re.match(r'\"\d+\-\d+\"\:(\d+)',num)
        result_train.append(int(values_train.group(1)))
    number_of_checkin_train.append(int(sum(result_train)))

# Splitting the testing file to generate number checked alone and summing it up
for line_test in test_Checkingfile:
    splitted_test = line_test.rstrip().split(',') 
    businessid_test.append(splitted_test[1])
    str_rev_test = re.sub(r'[\{\}]',"",splitted_test[0])
    split_test = str_rev_test.split(";")
    result_test = []
    for num in split_test:
        values_test = re.match(r'\"\d+\-\d+\"\:(\d+)',num)
        result_test.append(int(values_test.group(1)))
    number_of_checkin_test.append(int(sum(result_test)))

# Write to a CSV file, for training dataset
prediction_train = open("numberCheckinBusiness_trainDataset.csv","w")
prediction_train.write(str(("numberCheckin,BusinessID"+"\n")))
count_train = 0
for num in number_of_checkin_train:
   prediction_train.write((str(num)+","+str(businessid_train[count_train])+"\n"))
   count_train=count_train+1
prediction_train.close()

# Write to a CSV file, for testing dataset
prediction_test = open("numberCheckinBusiness_testDataset.csv","w")
prediction_test.write(str(("numberCheckin,BusinessID"+"\n")))
count_test = 0
for num in number_of_checkin_test:
   prediction_test.write((str(num)+","+str(businessid_test[count_test])+"\n"))
   count_test=count_test+1
prediction_test.close()

print len(number_of_checkin_train)
print len(businessid_train)
print len(number_of_checkin_test)
print len(businessid_test)
print "Completed"
