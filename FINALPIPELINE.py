import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.stats import f_oneway, kruskal, mannwhitneyu
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from antropy import sample_entropy
import pywt

# Define the paths to the CSV files
original_file_paths = {
    # Original datasets with channel swapping
    'baseline': [
        r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\ray\baseline\record-[2024.07.11-15.09.48].csv",
        r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\ray\baseline\record-[2024.07.11-15.18.47].csv",
        r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\ray\baseline\record-[2024.07.11-15.26.38].csv",
        r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\ray\baseline\record-[2024.07.11-15.41.48].csv",
        r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mama\baseline\record-[2024.07.09-15.59.14].csv",
        r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mama\baseline\record-[2024.07.09-16.08.42].csv",
        r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mama\baseline\record-[2024.07.09-16.18.56].csv",
        r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mama\baseline\record-[2024.07.09-15.41.02].csv",
        r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mama\baseline\record-[2024.07.09-15.50.18].csv",
        r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata\baseline\record-[2024.07.23-11.06.08].csv",
        r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata\baseline\record-[2024.07.23-11.15.22].csv",
        r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata\baseline\record-[2024.07.23-11.25.40].csv",
        r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata\baseline\record-[2024.07.23-10.47.47].csv",
        r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata\baseline\record-[2024.07.23-10.57.54].csv"
    ],
    'letters': [
        r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\ray\pho\record-[2024.07.11-16.07.26].csv",
        r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\ray\pho\record-[2024.07.11-16.22.21].csv",
        r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\ray\pho\record-[2024.07.11-15.51.42].csv",
        r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mama\pho\record-[2024.07.09-18.14.23].csv",
        r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mama\pho\record-[2024.07.09-17.48.34].csv",
        r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mama\pho\record-[2024.07.09-17.56.39].csv",
        r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata\pho\record-[2024.07.23-13.37.27].csv",
        r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata\pho\record-[2024.07.23-12.59.15].csv",
        r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata\pho\record-[2024.07.23-13.08.54].csv",
        r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata\pho\record-[2024.07.23-13.17.44].csv",
        r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata\pho\record-[2024.07.23-13.27.44].csv"
    ],
    'semantic': [
        r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\ray\semant\record-[2024.07.11-17.04.55].csv",
        r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\ray\semant\record-[2024.07.11-16.33.32].csv",
        r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\ray\semant\record-[2024.07.11-16.49.09].csv",
        r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\ray\semant\record-[2024.07.11-16.56.34].csv",
        r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mama\semant\record-[2024.07.09-17.21.16].csv",
        r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mama\semant\record-[2024.07.09-17.05.58].csv",
        r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata\semant\record-[2024.07.23-12.47.13].csv",
        r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata\semant\record-[2024.07.23-11.37.04].csv",
        r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata\semant\record-[2024.07.23-11.46.55].csv",
        r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata\semant\record-[2024.07.23-12.30.04].csv",
        r"C:\Users\brand\OneDrive - University College London\Thesis\brandon\mydata\semant\record-[2024.07.23-12.38.49].csv"
    ]
}

new_file_paths = {
    # New datasets without channel swapping
    'baseline': [
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

ch_names = [
    'AF7', 'FC3', 'FC1', 'F7', 'F9', 'FT9', 'C3', 'C1', 'FT7', 'O1',
    'Oz', 'CP3', 'CP1', 'T7', 'P5', 'PO3', 'F5', 'P1', 'F1', 'F3',
    'C5', 'CP5', 'FC5', 'P3', 'P7', 'PO7', 'TP7', 'TP9', 'P9', 'Fp1',
    'Fz', 'AF3'
]

sfreq = 512  # Sampling frequency

# Create the MNE Info object
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

# Initialize lists to store all epochs and event IDs
baseline_epochs = []
baseline_event_ids = []

letters_epochs = []
letters_event_ids = []

semantic_epochs = []
semantic_event_ids = []

# Definition of the event_ids
baseline_event_id = {'Up': 0, 'Right': 90, 'Down': 180, 'Left': 270}
letters_event_id = {'A': 0, 'C': 90, 'B': 180, 'D': 270}
semantic_event_id = {'Fly': 0, 'Future': 90, 'Dig': 180, 'Past': 270}

# Function to process files, including channel swapping if needed
def process_file(path, event_id, epochs_list, event_ids_list, swap_channels=False):
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
    noisy_channels = [ch for ch in raw.ch_names if np.std(raw.get_data(picks=[ch])) > 1e-6]

    raw.info['bads'] = flat_channels
    raw.drop_channels(flat_channels)

    if noisy_channels:
        raw_notch = raw.copy().notch_filter(freqs=np.arange(50, 241, 50), picks=noisy_channels)
        raw_filtered_low = raw_notch.filter(3., 20., picks=noisy_channels)
        raw_filtered_high = raw_notch.filter(80., 180., picks=noisy_channels)
        raw_combined_noisy = mne.concatenate_raws([raw_filtered_low, raw_filtered_high])
        raw_combined_noisy.apply_proj()
    else:
        raw_combined_noisy = raw.copy()

    raw_filtered_low = raw.copy().filter(3., 20., picks='eeg')
    raw_filtered_high = raw.copy().filter(80., 180., picks='eeg')

    raw_combined = mne.concatenate_raws([raw_filtered_low, raw_filtered_high])

    ica = mne.preprocessing.ICA(n_components=0.999999, random_state=97, max_iter=800)
    ica.fit(raw_combined)
    ica.exclude = [0, -1]  # Exclude the first component
    raw_combined.load_data()
    ica.apply(raw_combined)

    event_markers = [0, 90, 180, 270]
    events = np.array([[i, 0, int(event)] for i, event in enumerate(event_ids) if int(event) in event_markers], dtype=int)

    if events.shape[0] == 0 or events.shape[1] != 3:
        print(f"Warning: No valid events found in file {path}. Skipping this file.")
        return

    tmin, tmax = 0, 2.5  # Define time before and after the event in seconds
    epochs = mne.Epochs(raw_combined, events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=None, preload=True)

    epochs_list.append(epochs)
    event_ids_list.append(events)

# Process each file for each condition
for file_path in original_file_paths['baseline']:
    process_file(file_path, baseline_event_id, baseline_epochs, baseline_event_ids, swap_channels=True)

for file_path in original_file_paths['letters']:
    process_file(file_path, letters_event_id, letters_epochs, letters_event_ids, swap_channels=True)

for file_path in original_file_paths['semantic']:
    process_file(file_path, semantic_event_id, semantic_epochs, semantic_event_ids, swap_channels=True)

for file_path in new_file_paths['baseline']:
    process_file(file_path, baseline_event_id, baseline_epochs, baseline_event_ids, swap_channels=False)

for file_path in new_file_paths['letters']:
    process_file(file_path, letters_event_id, letters_epochs, letters_event_ids, swap_channels=False)

for file_path in new_file_paths['semantic']:
    process_file(file_path, semantic_event_id, semantic_epochs, semantic_event_ids, swap_channels=False)

baseline_epochs = mne.concatenate_epochs(baseline_epochs)
baseline_event_ids = np.concatenate(baseline_event_ids)

letters_epochs = mne.concatenate_epochs(letters_epochs)
letters_event_ids = np.concatenate(letters_event_ids)

semantic_epochs = mne.concatenate_epochs(semantic_epochs)
semantic_event_ids = np.concatenate(semantic_event_ids)

def compute_sample_entropy(epoch):
    return [sample_entropy(epoch[i]) for i in range(epoch.shape[0])]

def extract_features(epochs):
    psd_features, freqs = epochs.compute_psd(fmin=3.0, fmax=180.0).get_data(return_freqs=True)
    psd_features = 10 * np.log10(psd_features)
    
    wavelet_features = []
    for epoch in epochs.get_data():
        coeffs = pywt.wavedec(epoch, 'db4', level=4)
        features = np.concatenate([np.ravel(coeff) for coeff in coeffs], axis=-1)
        wavelet_features.append(features)
    wavelet_features = np.array(wavelet_features)
    
    sample_entropy_features = []
    for epoch in epochs.get_data():
        sample_entropy_features.append(compute_sample_entropy(epoch))
    
    sample_entropy_features = np.array(sample_entropy_features).reshape(epochs.get_data().shape[0], -1)
    
    combined_features = np.concatenate([
        psd_features.reshape(psd_features.shape[0], -1),
        wavelet_features,
        sample_entropy_features
    ], axis=1)
    
    return combined_features

baseline_labels = baseline_epochs.events[:, -1]
letters_labels = letters_epochs.events[:, -1]
semantic_labels = semantic_epochs.events[:, -1]

baseline_features = extract_features(baseline_epochs)
letters_features = extract_features(letters_epochs)
semantic_features = extract_features(semantic_epochs)

def train_and_evaluate(features, labels, title, event_id):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    svm = SVC(probability=True)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    voting_clf = VotingClassifier(estimators=[('knn', knn), ('svm', svm), ('rf', rf)], voting='soft')
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{title} Ensemble Classification Accuracy: {accuracy * 100:.2f}%")
    
    cv_scores = cross_val_score(voting_clf, features, labels, cv=5)
    print(f"{title} Cross-Validation Accuracy: {np.mean(cv_scores) * 100:.2f}%")
    print(f"Classification Report for {title}:\n")
    print(classification_report(y_test, y_pred, target_names=list(event_id.keys())))

    cm = confusion_matrix(y_test, y_pred, labels=list(event_id.values()))
    label_accuracies = cm.diagonal() / cm.sum(axis=1)
    for label, accuracy in zip(event_id.keys(), label_accuracies):
        print(f"{label} accuracy: {accuracy * 100:.2f}%")
    
    classifiers = {'KNN': knn, 'SVM': svm, 'Random Forest': rf}
    
    for clf_name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred_clf = clf.predict(X_test)
        accuracy_clf = accuracy_score(y_test, y_pred_clf)
        plt.figure()
        plt.bar(event_id.keys(), cm.diagonal() / cm.sum(axis=1))
        plt.ylim([0, 1])
        plt.title(f'{title} - {clf_name} Classification Accuracy: {accuracy_clf * 100:.2f}%')
        plt.ylabel('Accuracy')
        plt.show()
    
    return np.mean(cv_scores) * 100, label_accuracies * 100

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

participant_accuracies = {}

for participant, files in participants_data.items():
    print(f"\nProcessing data for {participant}...\n")
    
    participant_baseline_epochs = []
    participant_letters_epochs = []
    participant_semantic_epochs = []
    
    participant_baseline_event_ids = []
    participant_letters_event_ids = []
    participant_semantic_event_ids = []
    
    for file_path in files['baseline']:
        process_file(file_path, baseline_event_id, participant_baseline_epochs, participant_baseline_event_ids, swap_channels=False)
    
    for file_path in files['letters']:
        process_file(file_path, letters_event_id, participant_letters_epochs, participant_letters_event_ids, swap_channels=False)
    
    for file_path in files['semantic']:
        process_file(file_path, semantic_event_id, participant_semantic_epochs, participant_semantic_event_ids, swap_channels=False)
    
    participant_baseline_epochs = mne.concatenate_epochs(participant_baseline_epochs)
    participant_letters_epochs = mne.concatenate_epochs(participant_letters_epochs)
    participant_semantic_epochs = mne.concatenate_epochs(participant_semantic_epochs)
    
    participant_baseline_event_ids = np.concatenate(participant_baseline_event_ids)
    participant_letters_event_ids = np.concatenate(participant_letters_event_ids)
    participant_semantic_event_ids = np.concatenate(participant_semantic_event_ids)
    
    baseline_features = extract_features(participant_baseline_epochs)
    letters_features = extract_features(participant_letters_epochs)
    semantic_features = extract_features(participant_semantic_epochs)
    
    baseline_labels = participant_baseline_epochs.events[:, -1]
    letters_labels = participant_letters_epochs.events[:, -1]
    semantic_labels = participant_semantic_epochs.events[:, -1]
    
    baseline_accuracy, baseline_label_accuracies = train_and_evaluate(baseline_features, baseline_labels, f"{participant} Baseline", baseline_event_id)
    letters_accuracy, letters_label_accuracies = train_and_evaluate(letters_features, letters_labels, f"{participant} Letters", letters_event_id)
    semantic_accuracy, semantic_label_accuracies = train_and_evaluate(semantic_features, semantic_labels, f"{participant} Semantic", semantic_event_id)

    participant_accuracies[participant] = {
        'baseline': {
            'overall_accuracy': baseline_accuracy,
            'label_accuracies': dict(zip(baseline_event_id.keys(), baseline_label_accuracies))
        },
        'letters': {
            'overall_accuracy': letters_accuracy,
            'label_accuracies': dict(zip(letters_event_id.keys(), letters_label_accuracies))
        },
        'semantic': {
            'overall_accuracy': semantic_accuracy,
            'label_accuracies': dict(zip(semantic_event_id.keys(), semantic_label_accuracies))
        }
    }

for participant, wordsets in participant_accuracies.items():
    print(f"\nAccuracies for {participant}:")
    for wordset, accuracies in wordsets.items():
        print(f"\n{wordset.capitalize()} Overall Accuracy: {accuracies['overall_accuracy']:.2f}%")
        for label, accuracy in accuracies['label_accuracies'].items():
            print(f"{label} Accuracy: {accuracy:.2f}%")

baseline_accuracies = []
letters_accuracies = []
semantic_accuracies = []

for participant, wordsets in participant_accuracies.items():
    baseline_accuracies.append(wordsets['baseline']['overall_accuracy'])
    letters_accuracies.append(wordsets['letters']['overall_accuracy'])
    semantic_accuracies.append(wordsets['semantic']['overall_accuracy'])

anova_result = f_oneway(baseline_accuracies, letters_accuracies, semantic_accuracies)
print(f"ANOVA p-value: {anova_result.pvalue}")

if anova_result.pvalue < 0.05:
    data = pd.DataFrame({
        'Wordset': ['baseline'] * len(baseline_accuracies) + ['letters'] * len(letters_accuracies) + ['semantic'] * len(semantic_accuracies),
        'Accuracy': baseline_accuracies + letters_accuracies + semantic_accuracies
    })
    tukey_result = pairwise_tukeyhsd(data['Accuracy'], data['Wordset'])
    print(tukey_result)

kruskal_result = kruskal(baseline_accuracies, letters_accuracies, semantic_accuracies)
print(f"Kruskal-Wallis p-value: {kruskal_result.pvalue}")

if kruskal_result.pvalue < 0.05:
    pairwise_comparisons = data.groupby('Wordset')['Accuracy'].apply(lambda x: x.tolist())
    pairwise_comparisons = {k: v for k, v in pairwise_comparisons.items()}
    for ws1 in pairwise_comparisons:
        for ws2 in pairwise_comparisons:
            if ws1 != ws2:
                _, p_val = mannwhitneyu(pairwise_comparisons[ws1], pairwise_comparisons[ws2])
                print(f"Pairwise comparison between {ws1} and {ws2}: p-value = {p_val}")

# Function to plot spectrograms
#def plot_spectrogram(epoch_data, ch_idx, title):
    #f, t, Sxx = spectrogram(epoch_data[ch_idx, :], sfreq, nperseg=128, noverlap=64)
    #plt.figure()
    #plt.pcolormesh(t, f, np.log(Sxx))
    #plt.ylabel('Frequency [Hz]')
    #plt.xlabel('Time [sec]')
    #plt.title(title)
    #plt.colorbar(label='Log Power')
    #plt.show()

# Plot spectrograms for each participant, wordset, and electrode
#electrode_indices = {'F7': ch_names.index('F7'), 'P5': ch_names.index('P5')}

#for participant, files in participants_data.items():
    #for condition, epochs in zip(['baseline', 'letters', 'semantic'], [participant_baseline_epochs, participant_letters_epochs, participant_semantic_epochs]):
        #for event, label in zip([0, 90, 180, 270], ['Up', 'Right', 'Down', 'Left']):
            #for electrode, idx in electrode_indices.items():
                #epoch_data = epochs.copy().pick_channels([electrode]).get_data()[0]
                #plot_spectrogram(epoch_data, idx, f'{participant} - {condition.capitalize()} - {label} - {electrode}')

# Plot topomap for Participant 3
#times = [0.0, 0.5, 1, 1.5, 2, 2.5]
#for condition, epochs in zip(['baseline', 'letters', 'semantic'], [participant_baseline_epochs, participant_letters_epochs, participant_semantic_epochs]):
    #for event, label in zip([0, 90, 180, 270], ['Up', 'Right', 'Down', 'Left']):
        #epochs.copy().average().plot_topomap(times=times, ch_type='eeg', title=f'Participant 3 - {condition.capitalize()} - {label}')
