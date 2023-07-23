import streamlit as st
import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

col_logo, col_name = st.columns([20, 1])

# Company logo (replace 'logo.png' with the actual path to your logo image)
logo_path = 'Black & White Minimalist Business Logo.png'
col_logo.image(logo_path, use_column_width=True)

# Company name


st.title("EEG Signals Segmentation and Classification for Schizophrenia")
# st.write("**Our Model Gives an Accuracy of 99.26%**")

model = load_model('model5.h5')
uploaded_file = st.file_uploader("Choose a .edf file to upload", type=["edf"], key="file")

if uploaded_file is not None:
    file_name = os.path.basename(uploaded_file.name)
    st.write(f"File Name: {file_name}")
    st.write(f"File Size: {uploaded_file.size} bytes")

    with open(file_name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    raw = mne.io.read_raw_edf(file_name, preload=True)
    eeg_data = raw.get_data()
    channel_names = raw.ch_names

    # Define the frequency bands of interest
    frequency_bands = {'delta': (0.5, 4),
                       'theta': (4, 8),
                       'alpha': (8, 12),
                       'beta': (12, 30),
                       'gamma': (30, 100)}

    # Initialize a dictionary to store the component signals
    component_signals = {}

    # Extract component signals for each frequency band
    for band, (fmin, fmax) in frequency_bands.items():
        # Apply frequency filtering
        filtered_data = raw.copy().filter(fmin, fmax, fir_design='firwin')

        # Extract filtered EEG data
        filtered_eeg_data = filtered_data.get_data()

        # Store the component signal for the current band
        component_signals[band] = filtered_eeg_data

    # Print the shape of each component signal
    for band, signal in component_signals.items():
        print(f'{band} component signal shape: {signal.shape}')

    # Access the component signals from the dictionary
    alpha_signal = component_signals['alpha']
    beta_signal = component_signals['beta']
    gamma_signal = component_signals['gamma']
    delta_signal = component_signals['delta']
    theta_signal = component_signals['theta']

    # Combine the component signals column-wise
    combined_array = np.concatenate((alpha_signal, gamma_signal, theta_signal))
    combined_array = np.transpose(combined_array)

    combined_array = (combined_array - np.mean(combined_array)) / np.std(combined_array)

    X = combined_array[:6250]
    X = X.reshape([1, 6250, 57])

    Y = model.predict(X)
    ans = Y[0][0] * 100

    st.write("There is " + str(ans) + "% probability that this person suffers from " + "**Schizophrenia**" + ".")

    if (ans > 50):
        st.write("**We Recommend seeing a Doctor Immediately.**")

    os.remove(file_name)