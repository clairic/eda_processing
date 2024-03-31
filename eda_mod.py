import pandas as pd
from datetime import datetime, timedelta
import neurokit2 as nk
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import numpy as np

# Load EDA data 
eda_data = pd.read_csv('EDA.csv')
eda_signal = eda_data.iloc[:, 0]  # selecting the first column of the csv file
sampling_rate = 4

# Butterworth low-pass filtering 
def butter_lp_filter(data, cutoff, fs, order):
    nyq_f = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq_f
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# High-pass filtering to decompose the signal into tonic and phasic components
def highpass_filter(data, cutoff, fs, order):
    nyq_f = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq_f
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    z = filtfilt(b, a, data)
    return z

# Low-pass filter parameters
order_lp = 4
fs = 4  # sample rate 
cutoff_lp = 1  # desired cutoff frequency of the filter in Hz

# Applying the low-pass filter
filtered_signal = butter_lp_filter(eda_signal, cutoff_lp, fs, order_lp)

# High-pass filter parameters
order_hp = 4
cutoff_hp = 0.05  # Desired cutoff frequency of the high-pass filter, Hz

# Applying the high-pass filter and getting the phasic component
phasic_component = highpass_filter(filtered_signal, cutoff_hp, fs, order_hp)

# Getting the tonic component of the signal
tonic_component = filtered_signal - phasic_component

# Apply low-pass filter to the original EDA signal to get the SCL
cutoff_for_scl = 0.1  # Example low-pass filter cutoff frequency for SCL
order_for_scl = 4     # Filter order

scl_component = butter_lp_filter(eda_signal, cutoff_for_scl, sampling_rate, order_for_scl)

# Smooth the SCL signal
window_size = sampling_rate * 5  # Example: 5-second window for moving average
scl_smoothed = np.convolve(scl_component, np.ones(window_size) / window_size, mode='valid')

# Find SCR features
def find_scr_features(phasic_component, sampling_rate):
    scr_info = nk.eda_findpeaks(phasic_component, sampling_rate=sampling_rate)
    return scr_info

scr_features = find_scr_features(phasic_component, sampling_rate)

# Manually find SCR onsets
def find_scr_onsets(phasic_component, scr_features):
    onsets = []
    for peak in scr_features['SCR_Peaks']:
        onset = np.where(phasic_component[:peak] < phasic_component[peak]*0.1)[0][-1]
        onsets.append(onset)
    return onsets

scr_features['SCR_Onsets'] = find_scr_onsets(phasic_component, scr_features)

# Manually find SCR recovery
def find_scr_recovery(phasic_component, scr_features):
    recovery = []
    half_recovery = []
    for peak in scr_features['SCR_Peaks']:
        # Find the point where the phasic component drops to 10% of the peak after the peak
        recover_point = np.where(phasic_component[peak:] < phasic_component[peak]*0.1)[0]
        if len(recover_point) > 0:
            recover_point = recover_point[0] + peak
            half_recover_point = np.where(phasic_component[peak:] < phasic_component[peak]*0.5)[0]
            if len(half_recover_point) > 0:
                half_recover_point = half_recover_point[0] + peak
            else:
                half_recover_point = peak  # If no half recovery point is found, set it to the peak
        else:
            recover_point = peak  # If no recovery point is found, set it to the peak
            half_recover_point = peak  # If no half recovery point is found, set it to the peak
        recovery.append(recover_point)
        half_recovery.append(half_recover_point)
    return recovery, half_recovery

scr_features['SCR_Recovery'], scr_features['SCR_Half_Recovery'] = find_scr_recovery(phasic_component, scr_features)

# Generate timestamps
start_time = datetime.now()
time_stamps = [start_time + timedelta(seconds=i) for i in range(len(eda_signal))]

eda_data_with_features = pd.DataFrame({
    'Timestamp': time_stamps,
    'Raw EDA': eda_signal,
    'Cleaned EDA': filtered_signal,
    'SCR_Onsets': [1 if i in scr_features['SCR_Onsets'] else 0 for i in range(len(eda_signal))],
    'SCR_Peaks': [1 if i in scr_features['SCR_Peaks'] else 0 for i in range(len(eda_signal))],
    'SCR_Recovery': [1 if i in scr_features['SCR_Recovery'] else 0 for i in range(len(eda_signal))],
    'SCR_Half_Recovery': [1 if i in scr_features['SCR_Half_Recovery'] else 0 for i in range(len(eda_signal))]
})

plt.figure(figsize=(12, 8))

# Original EDA
plt.subplot(4, 1, 1)
plt.plot(eda_signal, label='Original EDA')
plt.title('Original EDA')
plt.xlabel('Samples')
plt.ylabel('EDA Amplitude')
plt.legend()

# Filtered EDA
plt.subplot(4, 1, 2)
plt.plot(filtered_signal, label='Filtered EDA', color='blue')
plt.title('Filtered EDA')
plt.xlabel('Samples')
plt.ylabel('EDA Amplitude')
plt.legend()

# Phasic Component
plt.subplot(4, 1, 3)
plt.plot(phasic_component, label='Phasic Component', color='red')
plt.scatter(scr_features['SCR_Peaks'], phasic_component[scr_features['SCR_Peaks']], color='blue', label='SCR Peaks')
for peak, onset, recovery, half_recovery in zip(scr_features['SCR_Peaks'], scr_features['SCR_Onsets'], scr_features['SCR_Recovery'], scr_features['SCR_Half_Recovery']):
    plt.plot([peak, peak], [0, phasic_component[peak]], 'b--')  # Vertical line from peak
    plt.plot([onset, peak], [phasic_component[onset], phasic_component[onset]], 'g-')  # Line from onset to peak
    plt.plot([half_recovery, peak], [phasic_component[half_recovery], phasic_component[half_recovery]], 'm-')  # Line from half recovery to peak
plt.title('Phasic Component with SCR Features')
plt.xlabel('Samples')
plt.ylabel('EDA Amplitude')
plt.legend()

# Tonic Component
plt.subplot(4, 1, 4)
plt.plot(tonic_component, label='Tonic Component', color='green')
plt.title('Tonic Component')
plt.xlabel('Samples')
plt.ylabel('EDA Amplitude')
plt.legend()

plt.tight_layout()

# Save the plot as PNG
plt.savefig('EDA_plot.png', format='png')

# Save the data with features
eda_data_with_features.to_csv('Modified_EDA.csv', index=False)

plt.show()
