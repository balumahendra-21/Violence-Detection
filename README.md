🛡️ Violence Detection System for Videos
====================================

An end-to-end deep learning system for detecting violent scenes in videos,
implemented using TensorFlow/Keras for modeling and Streamlit for the user interface.

🎯 Purpose
----------

This project automatically detects violence in videos using a trained deep learning model.
Users can upload a video, and the system will analyze its frames to predict whether violence is present.

🧠 Core Components
------------------

A. Model Training (mobilenetv21-model-checkpoint.ipynb)

- Data Preparation:
  - Videos are split into frames using OpenCV.
  - Frames are augmented (flipping, zooming, brightness, rotation) to improve model robustness.
  - Frames are resized to 128x128 pixels and normalized.

- Dataset:
  - Uses the "Real Life Violence Dataset" with two classes: "Violence" and "NonViolence".
  - 700 videos are sampled due to memory constraints.

- Model Architecture:
  - Based on MobileNetV2 (transfer learning).
  - The base model is frozen; a dense layer with sigmoid activation is added for binary classification.

- Training:
  - Uses callbacks for early stopping, learning rate scheduling, and model checkpointing.
  - Trains for up to 50 epochs.
  - Achieves high accuracy (~95%) and low loss on both training and validation sets.

- Evaluation:
  - Confusion matrix and classification report show strong precision, recall, and F1-score.
  - Model is saved as `my_modelnew.keras` (and previously as `modelnew.h5`).

B. Inference & User Interface (a.py)

- Streamlit App:
  - Users can upload videos via a web interface.
  - The uploaded video is processed frame by frame.
  - Each frame is resized and normalized before being passed to the trained model.
  - The model predicts violence probability for each frame.
  - Results are displayed on the frame (red text for violence, green for non-violence).
  - A summary message ("Violence Detected!" or "No Violence Detected.") is shown based on predictions.

🖥️ Workflow
-----------

Training Phase:
- Prepare and augment video frames.
- Train MobileNetV2-based model.
- Evaluate and save the best model.

Inference Phase:
- User uploads a video via Streamlit.
- Each frame is analyzed by the model.
- Results are displayed visually and as a summary.

📊 Performance
--------------

- Accuracy: ~95% on test set.
- Confusion Matrix: Shows high correct prediction rates for both classes.
- Classification Report: High precision, recall, and F1-score for violence and non-violence.

📁 Files Overview
-----------------

- mobilenetv21-model-checkpoint.ipynb: Model training, evaluation, and saving.
- a.py: Streamlit app for video upload and violence detection.
- modelnew.h5 / my_modelnew.keras: Saved trained models.
- Video/image files: Used for testing and demonstration.

🧰 Technologies Used
--------------------

- TensorFlow/Keras: Deep learning model.
- OpenCV: Video and image processing.
- Streamlit: Web-based user interface.
- imgaug: Data augmentation.
- scikit-learn: Data splitting and evaluation metrics.

⚙️ Installation Requirements
----------------------------

To run this project, install the following Python packages:

- Python >= 3.7
- TensorFlow >= 2.8
- Keras (included with TensorFlow)
- OpenCV-Python
- Streamlit
- imgaug
- scikit-learn
- numpy
- pandas
- matplotlib

You can install all dependencies using pip:

  $ pip install -r requirements.txt

Alternatively, install manually:

  $ pip install tensorflow opencv-python streamlit imgaug scikit-learn numpy pandas matplotlib

🚀 How to Use
-------------

- Train the model (if not already trained) using the notebook:

**  $ jupyter notebook mobilenetv21-model-checkpoint.ipynb **

- Run the Streamlit app:

**  $ streamlit run a.py **

- Upload a video in the web interface.

- View results: The app displays annotated frames and a summary of violence detection.

Summary
-------

This project provides a complete solution for detecting violence in videos using deep learning,
with a user-friendly web interface for real-time inference. The model is robust, accurate,
and suitable for practical deployment.

