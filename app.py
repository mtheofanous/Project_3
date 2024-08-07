import streamlit as st
import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import cv2
import joblib
import mediapipe as mp
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

from data import visualizaciones_datasets, show_random_image, show_image, load_gesture
from metrics import load_model_and_label_encoder, load_test_data, plot_confusion_matrix, generate_classification_report
import av
import subprocess
import requests

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set environment variables to suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

path = './sources/'

st.set_page_config(page_title = "ASL_Gesture_Recognition_App",
                   page_icon = "hand",
                   layout = 'centered',
                   initial_sidebar_state = 'expanded')

def main():
    
    st.title("ASL Gesture Recognition App.")

    # st.sidebar.success("Navigation")
    menu = ["Home", "Metrics and Visualization", "Real_time_Recognition"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("This web app recognizes American Sign Language (ASL) gestures using hand landmarks.")

        image_path = os.path.join(path, "homepage_image.jpg") 
        st.image(image_path, use_column_width=True, caption="American Sign Language (ASL) Gestures", width=400)


        st.write("""
            ## Overview
            American Sign Language (ASL) is a complete language that uses gestures and visual signs to communicate with people who are deaf or hard of hearing. ASL has its own grammar and syntax, making it a fully-fledged language in its own right.

            ## Dataset
            The dataset used for this project is sourced from Kaggle’s ASL Alphabet dataset. It contains:
            - **Training Data**: 87,000 images of ASL gestures, each of size 200x200 pixels.
            
            - **Categories**: 29 labeled categories including letters A-Z and additional gestures like "del", "nothing", and "space".
            """)
        
        # button for load random image
        if st.button("Show Random Image"):
            load_gesture(path=path + 'asl_alphabet_test')

        st.write("""
            ## Approaches Considered
            We considered three potential approaches to solve this problem:
            1. **Using the Whole Image**: This approach involves training a model on the entire image of the gesture.
            2. **Cropping the Image to Only the Hand**: Here, we would preprocess the images to isolate and use only the hand region.
            3. **Extracting Hand Landmarks**: This approach focuses on extracting and using the hand landmarks for classification.

            ## Chosen Approach
            After evaluating the pros and cons of each method, we decided to pursue the third approach: extracting hand landmarks. This decision was based on several factors:
            - **Dimensionality Reduction**: By focusing on hand landmarks, we reduce the size of our input data significantly, which simplifies the model and speeds up the training process.
            - **Increased Robustness**: Hand landmarks provide a more invariant and robust representation of gestures compared to raw pixel data, which can be affected by variations in lighting, background, and other noise.
            - **Relevance**: Hand landmarks contain the essential information needed to distinguish between different ASL gestures, making them an efficient choice for classification.
        """)
        if st.button("Hand Landmarks Example"):
    
            image_path = os.path.join(path, "21-3D-hand-landmarks-localized-by-MediaPipe-hand-tracking-model.png") 
            st.image(image_path, use_column_width=True, width=400)

    elif choice == "Metrics and Visualization":
        st.header("Metrics and Visualization")
        # Load the dataset
        csv_path = os.path.join(path, "hand_landmarks_augment.csv")
        df = pd.read_csv(csv_path)
        
        data_choice = st.sidebar.selectbox("Data Analysis and Metrics", 
                                           ["Show Dataset", 
                                            "Label Distribution", 
                                            "Show hand landmarks",
                                            "Confusion Matrix", "Classification Report"])

        if data_choice == "Show Dataset":
            st.write("### Dataset")
            st.dataframe(df)

        elif data_choice == "Label Distribution":
            st.write("### Label Distribution")
            visualizaciones_datasets(df)

        elif data_choice == "Show hand landmarks":
            st.write("### Hand Gesture Image")
            st.sidebar.write("### Hand Gesture Image")
            letter = st.sidebar.text_input("Enter a letter (A-Z):", 'B')
            if letter:
                show_random_image(letter, data_dir=r'C:\Users\DELL\Desktop\Project_3\sources\asl_alphabet_test')
    
        elif data_choice == "Confusion Matrix":
            # Load the model and label encoder
            model, le = load_model_and_label_encoder()
            # Load the test data
            X_test, y_test = load_test_data()

            y_pred = model.predict(X_test)

            # Inverse transform the labels

            y_test_trans = le.inverse_transform(y_test)
            y_pred_trans = le.inverse_transform(y_pred)

            cm = confusion_matrix(y_test_trans, y_pred_trans)

            if cm.size == 0:
                st.error("Unable to compute confusion matrix.")
            else:
                st.write("### Confusion Matrix")
                fig = plot_confusion_matrix(cm)
                if fig:
                    st.pyplot(fig)


        elif data_choice == "Classification Report":

            # Load the model and label encoder
            model, le = load_model_and_label_encoder()
            # Load the test data
            X_test, y_test = load_test_data()

            y_pred = model.predict(X_test)

            generate_classification_report(model=model, le=le, y_test=y_test, y_pred=y_pred)

    elif choice == "Real_time_Recognition":

        st.title("Live-Streaming ASL Gesture Recognition App")
 
        st.sidebar.success("Use the image grid below to practice ASL gestures. The app will recognize your gestures in real-time.")
        image_path = os.path.join("sources", "image_grid.jpg")
        st.sidebar.image(image_path, caption="Image Grid", use_column_width=True)

        # Load the pre-trained model and label encoder
        model, le = load_model_and_label_encoder() 

        # Initialize mediapipe Hands
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        from mediapipe.python.solutions.drawing_utils import DrawingSpec

        # Customize the styles
        landmark_style = DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2)
        connection_style = DrawingSpec(color=(0, 0, 0), thickness=2)

        # Function to apply hand gesture recognition
        class VideoTransformer(VideoTransformerBase):
            def __init__(self):
                self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
                self.predictions = deque(maxlen=10) 
                self.last_predicted_gesture = "No gesture detected yet."
                self.gesture_history = []



            def transform(self, frame: av.VideoFrame):

                img = frame.to_ndarray(format= "bgr24") 

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
              
                # img_rgb = cv2.resize(img, (200, 200)) # new
                img_rgb.flags.writeable = False # prevent modification of the image data and improve performance
                results = self.hands.process(img_rgb)

                # # Process the frame to find hand landmarks
                # frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # results = self.hands.process(frame_rgb)

                H, W, _ = img.shape

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw the hand landmarks
                        mp_drawing.draw_landmarks(
                            img,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            landmark_style,
                            connection_style
                        )

                        # Prepare data for prediction
                        data = []
                        x_ = []
                        y_ = []

                        for lm in hand_landmarks.landmark:
                            x_.append(lm.x)
                            y_.append(lm.y)

                        for i in range(len(hand_landmarks.landmark)):
                            data.append(hand_landmarks.landmark[i].x - min(x_))
                            data.append(hand_landmarks.landmark[i].y - min(y_))

                        # Predict the gesture
                        prediction = model.predict([np.asarray(data)])
                        predicted_character = le.inverse_transform(prediction)[0]

                        # Smooth predictions
                        self.predictions.append(predicted_character)
                        smoothed_prediction = max(set(self.predictions), key=self.predictions.count)

                        # Update last predicted gesture
                        self.last_predicted_gesture = f'Predicted Gesture: {smoothed_prediction}'

                        # Draw a bounding box around the hand
                        x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
                        x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10
                        
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 4)

                        # Put the predicted gesture on the frame
                        cv2.putText(img, smoothed_prediction, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

                return av.VideoFrame.from_ndarray(img, format="bgr24")

        ctx = webrtc_streamer(key="streamer", video_frame_callback=VideoTransformer().transform, sendback_audio=False)
        if ctx.video_transformer is not None:
            st.write("### Predictions")
        
            # Display the last predicted gesture
            st.write(ctx.video_transformer.last_predicted_gesture)
        else:
            st.write("### Predictions")
            st.write("No gesture detected yet.")

        

if __name__ == "__main__":
    main()
    