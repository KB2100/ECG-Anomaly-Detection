import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = load_model('/content/drive/MyDrive/ptbxl/ecg_model.h5')

# Function to detect anomaly
def detect_anomaly(file):
    try:
        # Load CSV file
        data = pd.read_csv(file.name)
        
        # Convert DataFrame to NumPy array
        data = data.to_numpy()
        
        # Fix input shape (always 12 leads)
        if data.shape[1] < 12:
            missing_cols = 12 - data.shape[1]
            zero_padding = np.zeros((data.shape[0], missing_cols))
            data = np.hstack((data, zero_padding))
        elif data.shape[1] > 12:
            data = data[:, :12]
        
        # Apply Min-Max Scaling to normalize data between 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data)
        
        # Reshape the data to (1, 1000, 12)
        signal_data = np.expand_dims(data, axis=0)
        
        # Make Prediction
        prediction = model.predict(signal_data)
        anomaly_score = prediction[0][0]
        
        # Apply a new threshold to catch even small anomalies
        if anomaly_score < 0.4:
            result = "✅ Normal ECG"
        elif 0.4 <= anomaly_score < 0.6:
            result = "⚠ Borderline Anomaly ECG"
        else:
            result = "❌ Anomalous ECG"
        
        # Plot the ECG Waveform
        plt.figure(figsize=(10, 4))
        for i in range(12):
            plt.plot(data[:, i], label=f'Lead {i+1}')
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title("ECG Waveform")
        plt.legend(loc="upper right")
        plot_path = "/content/ecg_plot.png"
        plt.savefig(plot_path)
        plt.close()
        
        return result, plot_path
    except Exception as e:
        return f"Error: {str(e)}", None

# Gradio Interface
interface = gr.Interface(
    fn=detect_anomaly,
    inputs=gr.File(label="Upload ECG CSV File", type='filepath'),
    outputs=[
        gr.Textbox(label="Prediction Result"),
        gr.Image(label="ECG Waveform")
    ]
)

interface.launch(share=True)
