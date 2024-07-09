import os
import random
import cv2
# import matplotlib.pyplot as plt
import mediapipe as mp
from PIL import Image
# import matplotlib
# matplotlib.use('TkAgg')  # Use an interactive backend
import matplotlib.pyplot as plt
import streamlit as st

# Set environment variables to suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize MediaPipe Hands and Drawing Utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

# Define the styles for drawing
landmark_style = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
connection_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)

def visualizaciones_datasets(df):

    labels = df['label'].value_counts().reset_index()
    labels.columns = ['label', 'count']
    labels['percentage'] = (labels['count'] / len(df)) * 100
    labels = labels.sort_values('count', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.pie(labels['percentage'], labels=labels['label'], autopct='%1.1f%%', startangle=140)
    plt.title('Labels Balance (Percentage)', loc='left')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.axis('equal')
    st.pyplot(fig)
def show_image(image_path):

    image = Image.open(image_path)
    st.image(image, caption='Uploaded Image', use_column_width=True)

def load_gesture(gesture='random', path='data_path_1'):

    # List the folders (gestures) in the dataset
    gestures = os.listdir(path)

    # Verify if the gesture is in the dataset
    if len(gesture) == 1:
        gesture = gesture.upper()
    if (gesture not in gestures) and (gesture != 'random'):
        print(f"Gesture {gesture} not found in dataset")
        print("Available gestures are: ", gestures)
        return None
    else:
        if gesture == 'random':
            gesture = random.choice(gestures)
            print(f"Random gesture selected: {gesture}")

        # There should be only one image per gesture folder
        image_path = os.path.join(path, gesture, os.listdir(os.path.join(path, gesture))[0])
        show_image(image_path)
        return image_path

def show_random_image(letter, data_dir='data_path'):

    letter_dir = os.path.join(data_dir, letter)
    
    if not os.path.exists(letter_dir):
        print(f"Directory for letter '{letter}' does not exist.")
        return

    images = [f for f in os.listdir(letter_dir) if f.endswith(('jpg', 'jpeg', 'png'))]
    
    if not images:
        print(f"No images found in the directory for letter '{letter}'.")
        return

    img_path = os.path.join(letter_dir, images[0])
    
    # Read and process the image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img_rgb,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=landmark_style,
                connection_drawing_spec=connection_style
            )
    
    # Convert RGB to BGR for displaying with OpenCV
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    # Display image using Matplotlib
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.title(f"Image from letter '{letter}'")
    plt.axis('off')
    
    st.pyplot(fig)
