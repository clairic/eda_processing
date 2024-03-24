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

#segmenting with a 30-second non overlapping window
window_length = 30 #in seconds
window_length_samples = window_length * sampling_rate 

#list to store segments
seg = []
#list to store results
filtered_segments = []
phasic_components = []
tonic_components = []
scl_components = []

#filtered signal - butterworth low pass filtering 
def butter_lp_filter(data, cutoff, fs, order):
    nyq_f = 0.5*fs #Nyquist Frequency
    normal_cutoff = cutoff/nyq_f
    b, a = butter(order, normal_cutoff, btype = 'low', analog=False)
    y = filtfilt(b,a,data)
    return y 

#decomposing signal into tonic and phasic components
def butter_hp_filter(data, cutoff, fs, order):
    nyq_f=0.5*fs #Nyquist Frequency
    normal_cutoff = cutoff/nyq_f
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    z = filtfilt(b, a, data)
    return z

#directory of the folders where the segments and the plots will be saved 
parent_dir = "Segments"
plot_dir = "EDA_plots"
eda_chars_plots_dir = "EDA_chars_plots"
path = os.path.join(parent_dir)
os.makedirs(path, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(eda_chars_plots_dir, exist_ok=True)

#Divide the signal into non-overlapping segments
for start in range(0, len(eda_signal), window_length_samples):
    end = start + window_length_samples
    if end <= len(eda_signal):
        segment = eda_signal[start:end]

        if len(segment) == window_length_samples:
            seg.append(segment)


            #filtering the segments
            #filter requirements for the low pass filter
            order_lp = 4
            fs = 4 #sample rate 
            cutoff_lp = 1 #desired cutoff frequency of the filter in Hz
            # Applying the low-pass filter
            filtered_signal = butter_lp_filter(segment, cutoff_lp, fs, order_lp)
            filtered_segments.append(filtered_signal)


            #decomposing the signal
            #filter requirements for the high pass filter
            order_hp = 4
            cutoff_hp = 0.05  # Desired cutoff frequency of the high-pass filter, Hz
            #applying the high-pass filter and getting the phasic component
            phasic_component = butter_hp_filter(filtered_signal, cutoff_hp, fs, order_hp)
            phasic_components.append(phasic_component)


            #getting the tonic component of the signal
            tonic_component = filtered_signal - phasic_component
            tonic_components.append(tonic_component)


            #getting the SCL from the raw signal
            cutoff_for_scl = 0.1  # Example low-pass filter cutoff frequency for SCL
            order_for_scl = 4     # Filter order
            # Apply low-pass filter to the original EDA signal
            scl_component = butter_lp_filter(segment, cutoff_for_scl, sampling_rate, order_for_scl)
            #smoothing the SCL signal
            window_size = sampling_rate * 5  # Example: 5-second window for moving average
            scl_smoothed = np.convolve(scl_component, np.ones(window_size) / window_size, mode='valid')
            scl_components.append(scl_component)


            #saving the filtered signal, phasic, tonic components in csv files
            for i, (original, filtered, phasic, tonic) in enumerate(zip(segment, filtered_segments, phasic_components, tonic_components)):
                
                segment_df = pd.DataFrame({
                'Original signal' : segment, 
                'Filtered EDA': filtered,
                'Phasic Component': phasic,
                'Tonic Component': tonic
                })
            
            filename = f"Segment_{i}_EDA.csv"
            complete_path = os.path.join(path, filename)

            segment_df.to_csv(complete_path, index=False)

            for i, scl_segment in enumerate (scl_components):
                plt.figure(figsize=(10,4))
                plt.plot(scl_segment, label='SCL Segment', color='purple')
                plt.title(f'Segment {i} - Skin Conductance Level (SCL)')
                plt.xlabel('Samples')
                plt.ylabel('SCL Amplitude')
                plt.legend()

                # Save the plot
                plt.savefig(f"{eda_chars_plots_dir}/Segment_{i}_SCL_plot.png")
                plt.close()

def find_scr_peaks(phasic_component, height=None, distance=None):
    peaks, _ = find_peaks(phasic_component, height=height, distance=distance)
    return peaks

#Saving the segment plots to png files
for i, (original, filtered, phasic, tonic) in enumerate(zip(seg, filtered_segments, phasic_components, tonic_components)):
    plt.figure(figsize=(12, 8))

    # Original EDA
    plt.subplot(4, 1, 1)
    plt.plot(original, label='Original EDA')
    plt.title(f'Segment {i} - Original EDA')
    plt.xlabel('Samples')
    plt.ylabel('EDA Amplitude')

    # Filtered EDA
    plt.subplot(4, 1, 2)
    plt.plot(filtered, label='Filtered EDA', color='blue')
    plt.title('Filtered EDA')
    plt.xlabel('Samples')
    plt.ylabel('EDA Amplitude')

    # Phasic Component
    plt.subplot(4, 1, 3)
    plt.plot(phasic, label='Phasic Component', color='red')
    plt.title('Phasic Component')
    plt.xlabel('Samples')
    plt.ylabel('EDA Amplitude')

    # Tonic Component
    plt.subplot(4, 1, 4)
    plt.plot(tonic, label='Tonic Component', color='green')
    plt.title('Tonic Component')
    plt.xlabel('Samples')
    plt.ylabel('EDA Amplitude')

    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(f"{plot_dir}/Segment_{i}_plot.png")
    plt.close()