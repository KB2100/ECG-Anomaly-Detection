# ECG-Anomaly-Detection
# ECG Anomaly Detection with CNN-LSTM and Explainability using SHAP

## 📜 Project Overview
The **ECG Anomaly Detection** project is an advanced deep learning application aimed at detecting anomalies in Electrocardiogram (ECG) signals. This project employs a hybrid **CNN-LSTM (Convolutional Neural Network - Long Short-Term Memory)** model to analyze ECG waveforms and identify potential heart abnormalities. Additionally, the model's decision-making process is interpreted using **SHAP (SHapley Additive exPlanations)** to enhance explainability in predictions.

This project provides a robust approach to ECG anomaly detection by processing multi-lead ECG signal data and identifying abnormal patterns using deep learning models, combined with post-hoc interpretability through SHAP.


## 📊 Dataset Information
The dataset used in this project is sourced from the **PTB-XL ECG dataset**, which contains:
- **12-lead ECG waveforms**.
- **Multiple classes**: Normal, Myocardial Infarction, ST-T changes, Conduction Disturbance, Hypertrophy, etc.
- **CSV files** that capture time-series ECG signals for thousands of patients.

The dataset was pre-processed and cleaned to fit into the CNN-LSTM model with normalization and shape adjustments



## 💻 Technologies Used
The project leverages the following technologies:
- **Python 3.10+**
- **TensorFlow / Keras** (For CNN-LSTM model)
- **NumPy / Pandas / Matplotlib / Seaborn** (For data manipulation and visualization)
- **Gradio** (For deploying and testing the model with real-time predictions)
- **SHAP** (For model explainability)
- **Scikit-learn** (For preprocessing and evaluation)



## 📊 Data Preprocessing
The raw ECG signals were pre-processed using the following steps:
1. **Handling Missing Values**: Any missing signals were filled or dropped.
2. **Normalization**: Applied MinMaxScaler to scale data between 0 and 1.
3. **Lead Conversion**: Converted 12-lead signals into a single-channel by averaging.
4. **Reshaping**: Resized the input data into a 3D shape (samples, timesteps, features).
5. **Train-Test Split**: Split the data into 80% training and 20% testing.

The goal was to prepare the data in a suitable format to train the CNN-LSTM model.


## 🧠 Model Architecture
### CNN-LSTM Hybrid Model
The project uses a **Convolutional Neural Network (CNN)** for spatial feature extraction from the ECG signal, followed by an **LSTM (Long Short-Term Memory)** network to capture temporal patterns and dependencies in the ECG signal.

The architecture consists of:
- **3 Convolutional Layers** with Batch Normalization and ReLU activation.
- **MaxPooling** to reduce the dimensionality.
- **LSTM Layer** with 100 units to capture sequential data.
- **Dense Layer** with sigmoid activation for binary classification.

The combination of CNN and LSTM improves both feature extraction and sequential pattern recognition.


## 📊 Model Evaluation
The model was evaluated using the following metrics:
- **Accuracy**: Measures overall correctness.
- **Precision**: Measures the true positive rate.
- **Recall (Sensitivity)**: Measures the ability to detect anomalies.
- **F1 Score**: Harmonic mean of precision and recall.
- **ROC-AUC Curve**: Measures the overall ability of the model to discriminate between normal and anomalous ECGs.

The model achieved high performance on all metrics, confirming its suitability for anomaly detection in ECG data.


## 📊 Explainability using SHAP
One of the key features of this project is the implementation of **SHAP (SHapley Additive exPlanations)** to interpret the model's predictions. 

### Why Explainability Matters?
Since ECG anomaly detection can have critical healthcare implications, it is essential to understand *why* the model is making a certain prediction. **SHAP** helps to:
- Identify which segments of the ECG signal contributed most to the anomaly.
- Provide transparent explanations to medical practitioners.
- Enhance trust in AI-assisted medical predictions.

The output includes:
- **SHAP Summary Plot**: Visualizing the impact of each lead in the prediction.
- **SHAP Force Plot**: Demonstrating how each feature pushed the model's prediction.


## ✅ Gradio Interface for Real-time Prediction
The project also features a **Gradio interface** that allows users to:
1. Upload CSV files containing ECG signals.
2. Click **Submit** to process the data.
3. View the predicted result (Normal, Borderline Anomaly, or Anomalous ECG).
4. See the visualized ECG waveform.

The Gradio app makes it easier for healthcare professionals to test real-world ECG signals in real-time.

To launch the Gradio app, use:
```shell
!python app.py
```

---

## 📊 Model Deployment
The model was deployed using **Gradio Public URL** and also integrated into GitHub for demonstration. The deployment steps include:
1. **Training the model** in Google Colab.
2. **Saving the model** as `ecg_model.h5`.
3. **Deploying the model** using Gradio with real-time CSV uploads.


For permanent hosting, you can deploy it on Hugging Face Spaces or Render.

---

## 📂 Folder Structure
```plaintext
├── data
│   ├── ecg_data.csv
├── model
│   ├── ecg_model.h5
├── app.py
├── README.md
├── requirements.txt
```

---

## ⚙ Installation
To run this project locally, follow these steps:

1. **Clone the repository:**
```shell
git clone https://github.com/your-username/ECG-Anomaly-Detection.git
cd ECG-Anomaly-Detection
```

2. **Install the dependencies:**
```shell
pip install -r requirements.txt
```

3. **Run the Gradio Interface:**
```shell
python app.py
```

4. **Upload any ECG CSV file** for real-time anomaly detection.



## 🤝 Acknowledgements
This project is heavily inspired by medical research papers on ECG anomaly detection, and the use of CNN-LSTM and SHAP for explainability.
- PTB-XL ECG Dataset: [https://physionet.org/content/ptb-xl/1.0.1/](https://physionet.org/content/ptb-xl/1.0.1/)
- SHAP Documentation: [https://shap.readthedocs.io/](https://shap.readthedocs.io/)
- Gradio Documentation: [https://gradio.app/](https://gradio.app/)

Feel free to raise issues, contribute, or star the repository if you find this project helpful!


## 📩 Contact
For any queries or collaboration, please contact me via:
- **Email**: bhattkashish322@gmail.com


### 🚀 If you like this project, don't forget to ⭐ the repository and share it with others!

Thank you! ❤️

