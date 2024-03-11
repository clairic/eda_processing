import pandas as pd
from datetime import datetime,timedelta 

#Load EDA data 
eda_data = pd.read_csv('C:/Users/khale/OneDrive/Desktop/EDA.csv')
eda_meas = eda_data.iloc[:,0] #selecting the first column of the csv file

#Defining the start time and the sampling rate 
start_time = datetime(2024,3,1,11,19,20) #the start time of the eda file I was given
sampling_rate = 1/4 #we have a 4Hz sampling period

#Generating timestamps
time_stamps=[start_time + timedelta(seconds=i) for i in range(len(eda_data))]

#Add timestamps to the DataFrame
eda_data_with_timestamps = pd.DataFrame({
    'Timestamp' : time_stamps,
    'EDA' : eda_meas
})

#Save the new DataFrame to a new CSV file. 
eda_data_with_timestamps.to_csv('Modified_EDA.csv', index=False)