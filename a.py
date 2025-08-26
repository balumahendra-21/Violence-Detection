import streamlit as st
import numpy as np
import cv2
from keras.models import load_model # type: ignore
from collections import deque
import tempfile

# Set Streamlit theme
st.set_page_config(layout="wide", page_title="Violence Detection App", page_icon=":no_entry_sign:")

def print_results(video, limit=None):
    model = load_model('modelnew.h5')  # Update the path based on your directory structure
    Q = deque(maxlen=128)
    vs = cv2.VideoCapture(video)
    (W, H) = (None, None)
    violence_detected = False

    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            break
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        output = frame.copy()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (128, 128)).astype("float32")
        frame = frame.reshape(128, 128, 3) / 255

        preds = model.predict(np.expand_dims(frame, axis=0))[0]
        Q.append(preds)

        i = (preds > 0.50)[0]
        label = i

        text_color = (0, 255, 0)  # default: green
        if label:  # Violence prob
            text_color = (0, 0, 255)  # red
            violence_detected = True
        else:
            text_color = (0, 255, 0)

        text = "Violence: {}".format(label)
        FONT = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(output, text, (35, 50), FONT, 1.25, text_color, 3)

        st.image(output, channels="BGR")  # Display the output image in Streamlit

    vs.release()

    if violence_detected:
        st.success("Violence Detected!")
    else:
        st.info("No Violence Detected.")

# Streamlit interface
st.title("Violence Detection in Videos")
st.markdown("Upload a video to detect violence.")

uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    st.write("Video uploaded successfully!")
    print_results(tfile.name)