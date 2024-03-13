import pandas as pd
from datetime import datetime,timedelta 
import neurokit2 as nk
import matplotlib.pyplot as plt
import os

#Load EDA data 
eda_data = pd.read_csv('C:/Users/khale/OneDrive/Desktop/EDA.csv')
eda_signal = eda_data.iloc[:,0] #selecting the first column of the csv file
sampling_rate = 1000
signals, info = nk.eda_process(eda_signal, sampling_rate)

cleaned_signal = signals["EDA_Clean"]
features = [info['SCR_Onsets'], info['SCR_Peaks'], info['SCR_Recovery']]
plot = nk.events_plot(features, cleaned_signal, color=['red', 'blue', 'orange'])

# Defining the start time and generating timestamps
start_time = datetime(2024,3,1,11,19,20) #the start time of the eda file I had recorded

#Generating timestamps
time_stamps=[start_time + timedelta(seconds=i) for i in range(len(eda_data))]
data = nk.eda_phasic(nk.standardize(eda_signal), sampling_rate)
data["Raw EDA"] = eda_signal
data.plot()
plot = nk.eda_plot(signals)
plt.show()

eda_data_with_features = pd.DataFrame({
    'Timestamp': time_stamps,
    'Raw EDA': eda_signal,
    'CLeaned EDA':signals['EDA_Clean'], 
    'SCR Onsets': [0] * len(eda_signal),
    'SCR Peaks': [0] * len(eda_signal),
    'SCR Recovery': [0] * len(eda_signal) })


# Update SCR event columns based on indices in info dictionary
for onset in info['SCR_Onsets']:
    eda_data_with_features.at[onset, 'SCR Onsets'] = 1

for peak in info['SCR_Peaks']:
    eda_data_with_features.at[peak, 'SCR Peaks'] = 1

for recovery in info['SCR_Recovery']:
    eda_data_with_features.at[recovery, 'SCR Recovery'] = 1


#Save the new DataFrame to a new CSV file. 
eda_data_with_features.to_csv('Modified_EDA.csv', index=False)

from firebase_admin import credentials, initialize_app, storage
# Init firebase with your credentials
cred = credentials.Certificate('usagestats-d296b-93f481d3896b.json')
initialize_app(cred, {'storageBucket': 'usagestats-d296b.appspot.com'})

# Put your local file path 
fileName = "Modified_EDA.csv"
bucket = storage.bucket()
blob = bucket.blob(fileName)
blob.upload_from_filename(fileName)

#Opt : if you want to make public access from the URL
blob.make_public()

print("your file url", blob.public_url)