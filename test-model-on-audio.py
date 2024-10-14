import pandas as pd
import numpy as np
import os
import librosa
import soundfile as sf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import joblib


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def extract_features(data, sample_rate):
    result = np.array([])

    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))

    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))

    return result


def get_features(path):
    data, sample_rate = librosa.load(path, duration=30, offset=0.6)
    return extract_features(data, sample_rate)



encoder = joblib.load("encoder.pkl")

scaler = joblib.load("scaler.pkl")

def predict_emotion(test_audio_file_path):

    
    model = tf.keras.models.load_model("/content/emotion_classifier_model.h5")

    audio, sample_rate = librosa.load(test_audio_file_path, duration=30, offset=0.6)

    audio_features = extract_features(audio, sample_rate)
    audio_features = np.array(audio_features.reshape(1,-1))
    audio_features = scaler.transform(audio_features)
    audio_features = np.expand_dims(audio_features, axis=2)  

    audio_features = np.array(audio_features.reshape(1,-1))
    predictions = model.predict(audio_features)
    print(predictions)
    predicted_emotion_index = np.argmax(predictions)
    predicted_emotion = encoder.categories_[0][predicted_emotion_index]
    return predicted_emotion


test_audio_file_path = "/content/test_audio.wav"
predicted_emotion = predict_emotion(test_audio_file_path)
print("Predicted emotion for the audio file:", predicted_emotion)