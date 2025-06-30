# Waste-Classifier-Project
This project is a smart waste classification system that identifies whether a piece of waste is Biodegradable, Hazardous, or Recyclable using a Convolutional Neural Network (CNN).

# Overview
1. Dataset: Custom labeled waste image dataset with 3 categories.
 
2. Model: CNN with 3 Conv2D layers and softmax output.
 
3. Goal: Automate waste type detection to promote proper waste management and recycling.

# Performance Summary
1.Training Accuracy: 90%

2.Test Accuracy: 73%

3. Classification Report Highlights:


    Biodegradable:
   
      Precision: 70–75%
      Recall: ~72%
   
    Hazardous:
   
      Precision: 65–70%
      Recall: ~68%
   
    Recyclable:
   
      Precision: 75–78%
      Recall: ~74%


4.Macro F1-Score: ~72%

5.Weighted F1-Score: ~73%

 The model shows balanced performance across all three classes and serves as a reliable base model for waste sorting automation

## 🚀 Deployment

This project is deployed on [Streamlit Cloud](https://streamlit.io/cloud). Below are the deployment steps:

### 🛠️ Deployment Steps

1. **Uploaded the project to GitHub** 
2. Specified the Python version compatible with TensorFlow:
    ```
    python-3.10
    ```
3. **Used Google Drive + `gdown`** for large model file download:
   
4. **requirements.txt**
    ```
    streamlit==1.33.0
    tensorflow==2.15.0
    numpy
    Pillow
    gdown
    ```
5. Deployed the app on Streamlit Cloud

   # [Link to Deployed app](https://waste-classifier-gcfrdpqx9wkmuhnhhotyre.streamlit.app/)
