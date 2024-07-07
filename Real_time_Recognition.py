import cv2
import streamlit as st
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import DrawingSpec
import warnings
from collections import deque
import joblib
import os

import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes
import numpy as np
import av


# Set environment variables to suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# st.title("OpenCV Filters on Video Stream")

# filter = "none"

# def transform(frame: av.VideoFrame):
#     img = frame.to_ndarray(format="bgr24")

#     if filter == "blur":
#         img = cv2.GaussianBlur(img, (21, 21), 0)
#     elif filter == "canny":
#         img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
#     elif filter == "grayscale":
#         # We convert the image twice because the first conversion returns a 2D array.
#         # the second conversion turns it back to a 3D array.
#         img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
#     elif filter == "sepia":
#         kernel = np.array(
#             [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]
#         )
#         img = cv2.transform(img, kernel)
#     elif filter == "invert":
#         img = cv2.bitwise_not(img)
#     elif filter == "none":
#         pass

#     return av.VideoFrame.from_ndarray(img, format="bgr24")

# webrtc_streamer(
#     key="streamer",
#     video_frame_callback=transform,
#     sendback_audio=False
#     )

