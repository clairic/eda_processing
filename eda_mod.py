import pandas as pd
from datetime import datetime,timedelta 
import neurokit2 as nk
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import numpy as np  
import os

#Load EDA data 
eda_data = pd.read_csv('EDA.csv')
eda_signal = eda_data.iloc[:,0] #selecting the first column of the csv file
sampling_rate = 4
# Generate timestamps
start_time = datetime.now()
# Define a specific start time
start_time = datetime(2024, 3, 1, 11, 19, 20)
time_stamps = [start_time + timedelta(seconds=i/4) for i in range(len(eda_signal))]

#filtered signal - butterworth low pass filtering 
def butter_lp_filter(data, cutoff, fs, order):
    nyq_f = 0.5*fs #Nyquist Frequency
    normal_cutoff = cutoff/nyq_f
    b, a = butter(order, normal_cutoff, btype = 'low', analog=False)
    y = filtfilt(b,a,data)
    return y 

#decomposing signal into tonic and phasic components
def highpass_filter(data, cutoff, fs, order):
    nyq_f=0.5*fs #Nyquist Frequency
    normal_cutoff = cutoff/nyq_f
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    z = filtfilt(b, a, data)
    return z

#filtering the segments
#filter requirements for the low pass filter
order_lp = 4
fs = 4 #sample rate 
cutoff_lp = 1 #desired cutoff frequency of the filter in Hz
# Applying the low-pass filter
filtered_signal = butter_lp_filter(eda_signal, cutoff_lp, fs, order_lp)


#decomposing the signal
#filter requirements for the high pass filter
order_hp = 4
cutoff_hp = 0.05  # Desired cutoff frequency of the high-pass filter, Hz
#applying the high-pass filter and getting the phasic component
phasic_component = highpass_filter(filtered_signal, cutoff_hp, fs, order_hp)


#getting the tonic component of the signal
tonic_component = filtered_signal - phasic_component


#getting the SCL from the raw signal
cutoff_for_scl = 0.1  # Example low-pass filter cutoff frequency for SCL
order_for_scl = 4     # Filter order
# Apply low-pass filter to the original EDA signal
scl_component = butter_lp_filter(eda_signal, cutoff_for_scl, sampling_rate, order_for_scl)
#smoothing the SCL signal
window_size = sampling_rate * 5  # Example: 5-second window for moving average
scl_smoothed = np.convolve(scl_component, np.ones(window_size) / window_size, mode='valid')

def find_scr_peaks(phasic_component, height=None, distance=None):
    peaks, _ = find_peaks(phasic_component, height=0.2, distance=distance)
    return peaks

scr_peaks = find_scr_peaks(phasic_component, height=None, distance=None)

eda_data_with_features = pd.DataFrame({
    'Timestamp': [ts.strftime('%H:%M:%S') for ts in time_stamps],
    'Raw EDA': eda_signal,
    'Cleaned EDA': filtered_signal, 
    'SCR Peaks': [1 if i in scr_peaks else 0 for i in range(len(eda_signal))]
})

# Save the eda_data_with_features DataFrame to a new CSV file
eda_data_with_features.to_csv('Modified_EDA.csv', index=False)

#Plots
plt.figure(figsize=(12, 8))
# Original EDA
plt.subplot(4, 1, 1)
plt.plot(eda_signal, label='Original EDA')
plt.title('Original EDA')
plt.xlabel('Samples')
plt.ylabel('EDA Amplitude')

# Filtered EDA
plt.subplot(4, 1, 2)
plt.plot(filtered_signal, label='Filtered EDA', color='blue')
plt.title('Filtered EDA')
plt.xlabel('Samples')
plt.ylabel('EDA Amplitude')

# Phasic Component
plt.subplot(4, 1, 3)
scr_peaks = find_scr_peaks(phasic_component, height=None, distance=None)  # Adjust parameters as needed
plt.plot(scr_peaks, phasic_component[scr_peaks], 'bo', label='SCR Peaks')  # 'bo' marks the peaks with blue dots
plt.plot(phasic_component, label='Phasic Component', color='red')
plt.title('Phasic Component with SCR Peaks')
plt.xlabel('Samples')
plt.ylabel('EDA Amplitude')

# Tonic Component
plt.subplot(4, 1, 4)
plt.plot(tonic_component, label='Tonic Component', color='green')
plt.title('Tonic Component')
plt.xlabel('Samples')
plt.ylabel('EDA Amplitude')

plt.tight_layout()

# Save the plot as PDF
plt.savefig('EDA_plot.png', format='png')

plt.show()