import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# Define the paths to the CSV files
participants_data = {
    'Participant 1': {
        'baseline': [
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\ray\baseline\record-[2024.07.11-15.09.48].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\ray\baseline\record-[2024.07.11-15.18.47].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\ray\baseline\record-[2024.07.11-15.26.38].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\ray\baseline\record-[2024.07.11-15.41.48].csv"
        ],
        'letters': [
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\ray\pho\record-[2024.07.11-16.07.26].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\ray\pho\record-[2024.07.11-16.22.21].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\ray\pho\record-[2024.07.11-15.51.42].csv"
        ],
        'semantic': [
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\ray\semant\record-[2024.07.11-17.04.55].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\ray\semant\record-[2024.07.11-16.33.32].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\ray\semant\record-[2024.07.11-16.49.09].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\ray\semant\record-[2024.07.11-16.56.34].csv"
        ]
    },
    'Participant 2': {
        'baseline': [
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mama\baseline\record-[2024.07.09-15.59.14].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mama\baseline\record-[2024.07.09-16.08.42].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mama\baseline\record-[2024.07.09-16.18.56].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mama\baseline\record-[2024.07.09-15.41.02].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mama\baseline\record-[2024.07.09-15.50.18].csv"
        ],
        'letters': [
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mama\pho\record-[2024.07.09-18.14.23].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mama\pho\record-[2024.07.09-17.48.34].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mama\pho\record-[2024.07.09-17.56.39].csv"
        ],
        'semantic': [
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mama\semant\record-[2024.07.09-17.21.16].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mama\semant\record-[2024.07.09-17.05.58].csv"
        ]
    },
    'Participant 3': {
        'baseline': [
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata\baseline\record-[2024.07.23-11.06.08].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata\baseline\record-[2024.07.23-11.15.22].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata\baseline\record-[2024.07.23-11.25.40].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata\baseline\record-[2024.07.23-10.47.47].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata\baseline\record-[2024.07.23-10.57.54].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata2\baseline\record-[2024.07.30-12.36.55].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata2\baseline\record-[2024.07.30-12.56.34].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata2\baseline\record-[2024.07.30-11.52.56].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata2\baseline\record-[2024.07.30-12.00.32].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata2\baseline\record-[2024.07.30-12.22.09].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata2\baseline\record-[2024.08.02-11.04.54].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata2\baseline\record-[2024.08.02-11.28.13].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata2\baseline\record-[2024.08.02-11.41.16].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata2\baseline\record-[2024.08.02-15.05.10].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata2\baseline\record-[2024.08.02-15.11.43].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata2\baseline\record-[2024.08.02-15.28.24].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata2\baseline\record-[2024.08.02-14.58.32].csv"
        ],
        'letters': [
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata\pho\record-[2024.07.23-13.37.27].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata\pho\record-[2024.07.23-12.59.15].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata\pho\record-[2024.07.23-13.08.54].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata\pho\record-[2024.07.23-13.17.44].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata\pho\record-[2024.07.23-13.27.44].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata2\pho\record-[2024.07.30-13.45.17].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata2\pho\record-[2024.07.30-13.59.23].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata2\pho\record-[2024.07.30-14.15.06].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata2\pho\record-[2024.07.30-13.20.22].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata2\pho\record-[2024.07.30-13.34.28].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata2\pho\record-[2024.08.02-12.34.47].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata2\pho\record-[2024.08.02-12.57.03].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata2\pho\record-[2024.08.02-12.12.14].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata2\pho\record-[2024.08.02-15.53.16].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata2\pho\record-[2024.08.02-16.03.36].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata2\pho\record-[2024.08.02-16.21.11].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata2\pho\record-[2024.08.02-15.46.02].csv"
        ],
        'semantic': [
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata\semant\record-[2024.07.23-12.47.13].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata\semant\record-[2024.07.23-11.37.04].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata\semant\record-[2024.07.23-11.46.55].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata\semant\record-[2024.07.23-12.30.04].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata\semant\record-[2024.07.23-12.38.49].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata2\semant\record-[2024.07.30-15.13.04].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata2\semant\record-[2024.07.30-14.38.11].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata2\semant\record-[2024.07.30-14.48.25].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata2\semant\record-[2024.07.30-14.55.11].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata2\semant\record-[2024.07.30-15.01.51].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata2\semant\record-[2024.08.02-13.42.56].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata2\semant\record-[2024.08.02-13.56.18].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata2\semant\record-[2024.08.02-13.23.44].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata2\semant\record-[2024.08.02-16.57.10].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata2\semant\record-[2024.08.02-17.13.28].csv",
            r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata2\semant\record-[2024.08.02-16.40.08].csv"
        ]
    }
}

# Channel names and sampling frequency
ch_names = [
    'AF7', 'FC3', 'FC1', 'F7', 'F9', 'FT9', 'C3', 'C1', 'FT7', 'O1',
    'Oz', 'CP3', 'CP1', 'T7', 'P5', 'PO3', 'F5', 'P1', 'F1', 'F3',
    'C5', 'CP5', 'FC5', 'P3', 'P7', 'PO7', 'TP7', 'TP9', 'P9', 'Fp1',
    'Fz', 'AF3'
]
sfreq = 512  # Sampling frequency

# Create the MNE Info object
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

# Define event IDs
baseline_event_id = {'Up': 0, 'Right': 90, 'Down': 180, 'Left': 270}
letters_event_id = {'A': 0, 'C': 90, 'B': 180, 'D': 270}
semantic_event_id = {'Fly': 0, 'Future': 90, 'Dig': 180, 'Past': 270}

# Function to process files
def process_file(path, event_id, epochs_list, swap_channels=False):
    data = pd.read_csv(path)
    eeg_data = data.iloc[:, 2:34].values.T

    if swap_channels:
        eeg_data = np.vstack([eeg_data[16:], eeg_data[:16]])

    event_ids = data.iloc[:, 34].values
    event_ids = np.nan_to_num(event_ids, nan=-1)

    raw = mne.io.RawArray(eeg_data / 1e6, info)  # Assuming the data is in microvolts, convert to volts

    mapping = {'Oz': 'OZ', 'Fp1': 'FP1', 'Fz': 'FZ'}
    raw.rename_channels(mapping)

    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False)

    flat_channels = [ch for ch in raw.ch_names if np.std(raw.get_data(picks=[ch])) < 1e-12]
    raw.drop_channels(flat_channels)

    raw_filtered_low = raw.copy().filter(3., 20., picks='eeg')
    raw_filtered_high = raw.copy().filter(80., 180., picks='eeg')

    raw_combined = mne.concatenate_raws([raw_filtered_low, raw_filtered_high])

    events = np.array([[i, 0, int(event)] for i, event in enumerate(event_ids) if int(event) in event_id.values()], dtype=int)

    if events.shape[0] == 0 or events.shape[1] != 3:
        print(f"Warning: No valid events found in file {path}. Skipping this file.")
        return

    tmin, tmax = 0, 2.5  # Define time before and after the event in seconds
    epochs = mne.Epochs(raw_combined, events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=None, preload=True)

    epochs_list.append(epochs)

# Prepare to store epochs
baseline_epochs, letters_epochs, semantic_epochs = [], [], []

# Process each file for each participant
for participant, conditions in participants_data.items():
    for condition, file_paths in conditions.items():
        event_id = eval(f"{condition}_event_id")
        for file_path in file_paths:
            process_file(file_path, event_id, eval(f"{condition}_epochs"), swap_channels=False)

# Concatenate epochs for each condition
baseline_epochs = mne.concatenate_epochs(baseline_epochs)
letters_epochs = mne.concatenate_epochs(letters_epochs)
semantic_epochs = mne.concatenate_epochs(semantic_epochs)

# Spectrogram plotting function
def plot_spectrogram(epoch_data, ch_idx, title):
    f, t, Sxx = spectrogram(epoch_data[ch_idx, :], sfreq, nperseg=128, noverlap=64)
    plt.figure()
    plt.pcolormesh(t, f, np.log(Sxx))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(title)
    plt.colorbar(label='Log Power')
    plt.show()

# Plot spectrograms for each participant, wordset, and electrode
electrode_indices = {'F7': ch_names.index('F7'), 'P5': ch_names.index('P5')}

for participant, files in participants_data.items():
    for condition, epochs in zip(['baseline', 'letters', 'semantic'], [baseline_epochs, letters_epochs, semantic_epochs]):
        for event, label in eval(f"{condition}_event_id").items():
            for electrode, idx in electrode_indices.items():
                epoch_data = epochs[label].get_data()[0]
                plot_spectrogram(epoch_data, idx, f'{participant} - {condition.capitalize()} - {event} - {electrode}')

# Plot topomap for Participant 3
times = [0.0, 0.5, 1, 1.5, 2, 2.5]
for condition, epochs in zip(['baseline', 'letters', 'semantic'], [baseline_epochs, letters_epochs, semantic_epochs]):
    for event, label in eval(f"{condition}_event_id").items():
        epochs[label].average().plot_topomap(times=times, ch_type='eeg')
