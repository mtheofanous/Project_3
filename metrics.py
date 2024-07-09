import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import warnings

# Set environment variables to suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)

def load_model_and_label_encoder(model_dir="sources", model_filename="model_landmarks_augment.pkl", label_filename="label_encoder_augment.pkl"):
    model_path = os.path.join(model_dir, model_filename)
    label_path = os.path.join(model_dir, label_filename)
    
    model, le = None, None
    
    if os.path.isfile(model_path) and os.path.isfile(label_path):
        model = joblib.load(model_path)
        le = joblib.load(label_path)
        print("Model and label encoder loaded successfully.")
    else:
        if not os.path.isfile(model_path):
            print(f"Model file not found: {model_path}")
        if not os.path.isfile(label_path):
            print(f"Label encoder file not found: {label_path}")
    
    return model, le

def load_test_data(file_path='sources/hand_landmarks_augment_test.csv'):
    model, le = load_model_and_label_encoder()
    if not os.path.exists(file_path):
        return pd.DataFrame(), None  # Handle missing CSV file
    test_data = pd.read_csv(file_path)
    X_test = test_data.drop("label", axis=1)
    y_test = test_data["label"]
    y_test = le.transform(y_test)
    return X_test, y_test

def compute_confusion_matrix(model, X_test, y_test, classes):
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred, labels=classes)

# def generate_classification_report(model, X_test, y_test, classes):
#     y_pred = model.predict(X_test)
#     report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
#     return pd.DataFrame(report).transpose()

def plot_confusion_matrix(conf_matrix):
    if conf_matrix.size == 0:
        return None  # Handle empty confusion matrix
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    return fig

def generate_classification_report(model, y_test, y_pred, le):

    # Generate the classification report
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

    # Convert the report to a DataFrame
    report_df = pd.DataFrame(report).transpose()

    # Display the report in Streamlit
    st.write("### Classification Report")
    st.dataframe(report_df)





