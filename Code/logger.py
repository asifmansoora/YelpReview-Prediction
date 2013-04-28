import datetime

def logg_result(Regressor_Name,count,RMSLE, time_tak):
        result_log = open("result_log.csv","a")
        result_log.write(str("***************log*************************"+"\n"))
        result_log.write(str("Time logged :"+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+"\n"))
        result_log.write(str("Model Used : " +str(Regressor_Name+ "\n")))
        result_log.write(str("Number of samples: "+ str(count)+ "\n" ))
        result_log.write(str("Root Mean Square Log Error: " +str(RMSLE) + "\n"))
        result_log.write(str("Time taken: " + str(time_tak)+"\n"))
        result_log.close()