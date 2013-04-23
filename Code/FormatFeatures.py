
import re
import numpy as np



 # extract a specific column from the matrix
def column(matrix, i):
    return [row[i] for row in matrix]


# fill the null values in a column with the mean value
def fill_avg(colmn):
    mask = np.isnan(colmn)
    masked_arr = np.ma.masked_array(colmn,mask)
    mean_val = np.mean(masked_arr,axis=0)
    return masked_arr.filled(mean_val)