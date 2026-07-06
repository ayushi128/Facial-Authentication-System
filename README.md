## Facial Authentication System

A deep learning-based facial authentication system that combines computer vision, deep learning, and classical machine learning to provide secure real-time user authentication.

# Overview

This project authenticates users using facial recognition and grants access to a private encrypted workspace after successful verification.

The system uses MediaPipe for face detection, a fine-tuned ResNet50 model for feature extraction, and a stacking ensemble classifier for identity prediction. Authenticated users can securely store notes, which are encrypted using AES-128 encryption.

## Features

- Real-time webcam-based facial authentication
- Fine-tuned ResNet50 feature extraction
- Stacking ensemble classifier for identity prediction
- **5-frame stability verification to reduce false positive authentications**
- AES-128 encrypted private workspace
- Facial landmark visualization using MediaPipe
- User registration and authentication interface
- Comparative evaluation of 9 machine learning models

  
## Technologies Used

- Python
- PyTorch
- OpenCV
- MediaPipe
- scikit-learn
- NumPy
- Pandas
- Tkinter

## Experimental Results

- Evaluated 9 machine learning classifiers on facial embeddings extracted using a fine-tuned ResNet50 model.
- Compared model performance using training/testing accuracy, cross-validation, weighted F1 score, confusion matrix, FAR & FRR analysis, PCA variance, and overfitting analysis.
- Generated detailed comparative visualizations and evaluation reports (`model_comparison.png` and `model_results.csv`).
