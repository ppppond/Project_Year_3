import streamlit as st
from labeling import labeling_page
from train import train_page
from predict import predict_page
from webcam import webcam_page
from video_scan import video_scan_page

st.set_page_config(
    page_title="Mini Roboflow",
    layout="wide",
    page_icon="🧊",
)

from preview import preview_page  

menu = st.sidebar.radio(
    "เมนูใช้งาน",
    [
        "1. Labeling (วาดกรอบ)",
        "2. Preview Labeled Images",  
        "3. Train (สอน AI)",
        "4. Predict (ทำนาย)",
        "5. Webcam (กล้องสด)",
        "6. Video Scan (วิดีโอ)",
        
    ],
)

if "Labeling" in menu:
    labeling_page()
elif "Train" in menu:
    train_page()
elif "Predict" in menu:
    predict_page()
elif "Webcam" in menu:
    webcam_page()
elif "Video Scan" in menu:
    video_scan_page()
elif "Preview" in menu:
    preview_page()  
