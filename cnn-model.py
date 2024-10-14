# Paths for data
RAVDESS = "/content/datasets/ravdess-emotional-speech-audio/audio_speech_actors_01-24/"
Crema = "/content/datasets/cremad/AudioWAV/"
Tess = "/content/datasets/toronto-emotional-speech-set-tess/tess toronto emotional speech set data/TESS Toronto emotional speech set data/"
Savee = "/content/datasets/surrey-audiovisual-expressed-emotion-savee/ALL/"

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

import soundfile as sf
def addnoise(signal,noisefactor):
  noise=np.random.normal(0,signal.std(),signal.size)
  augmented=signal+noise*noisefactor
  return augmented

def dataaugment(dataset_path):
  for root,dirs,files in os.walk(dataset_path):
    for file in files:
      file_path = os.path.join(root, file)
      signal,sr=librosa.load(file_path,sr=None)
      augmented=addnoise(signal,0.01)
      newfile="augmented"+file
      file_path = os.path.join(root, newfile)
      sf.write(file_path,augmented,samplerate=sr)
dataaugment("/content/datasets/ravdess-emotional-speech-audio/audio_speech_actors_01-24/")
dataaugment("/content/datasets/cremad/AudioWAV/")
dataaugment("/content/datasets/toronto-emotional-speech-set-tess/tess toronto emotional speech set data/TESS Toronto emotional speech set data/")
dataaugment("/content/datasets/surrey-audiovisual-expressed-emotion-savee/ALL/")

def preprocess_audio(dataset_path):
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                audio, sr = librosa.load(file_path, sr=None)
                if len(audio) > sr * 30:
                    audio = audio[:sr * 30]
                    
                non_silent_intervals = librosa.effects.split(audio, top_db=50)
                trimmed_audio = np.concatenate([audio[start:end] for start, end in non_silent_intervals])

                sf.write(file_path, trimmed_audio, sr)


    print("preprocessed ", root)



preprocess_audio("/content/datasets/ravdess-emotional-speech-audio/audio_speech_actors_01-24/")
preprocess_audio("/content/datasets/cremad/AudioWAV/")
preprocess_audio("/content/datasets/toronto-emotional-speech-set-tess/tess toronto emotional speech set data/TESS Toronto emotional speech set data/")
preprocess_audio("/content/datasets/surrey-audiovisual-expressed-emotion-savee/ALL/")

# Get file paths and emotions for RAVDESS dataset
file_emotion = []
file_path = []
for actor in os.listdir(RAVDESS):
    actor_path = os.path.join(RAVDESS, actor)
    print(actor_path)
    for file in os.listdir(actor_path):
        part = file.split('.')[0].split('-')
        print(part)
        print(file_emotion)
        file_emotion.append(int(part[2]))
        file_path.append(os.path.join(actor_path, file))

# Create DataFrame for RAVDESS dataset
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
path_df = pd.DataFrame(file_path, columns=['Path'])
RAVDESS_df = pd.concat([emotion_df, path_df], axis=1)

# Convert emotions to labels


RAVDESS_df.Emotions.replace({
    1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
    5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'
}, inplace=True)


# Get file paths and emotions for CREMA dataset
file_emotion = []
file_path = []

for file in os.listdir(Crema):

    file_path.append(Crema + file)

    print(part)
    part=file.split('_')
    if part[2] == 'SAD':
        file_emotion.append('sad')
    elif part[2] == 'ANG':
        file_emotion.append('angry')
    elif part[2] == 'DIS':
        file_emotion.append('disgust')
    elif part[2] == 'FEA':
        file_emotion.append('fear')
    elif part[2] == 'HAP':
        file_emotion.append('happy')
    elif part[2] == 'NEU':
        file_emotion.append('neutral')
    else:
        file_emotion.append('Unknown')

# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Crema_df = pd.concat([emotion_df, path_df], axis=1)



file_emotion = []
file_path = []

for dir in os.listdir(Tess):
    directories = os.listdir(Tess + dir)
    for file in directories:
        print(file)
        part = file.split('.')[0]
        part = part.split('_')[2]
        if part=='ps':
            file_emotion.append('surprise')
        else:
            file_emotion.append(part)
        file_path.append(Tess + dir + '/' + file)

# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Tess_df = pd.concat([emotion_df, path_df], axis=1)


file_emotion = []
file_path = []

for file in os.listdir(Savee):
    print(file)
    file_path.append(Savee + file)
    part = file.split('_')[1]
    ele = part[:-6]
    if ele=='a':
        file_emotion.append('angry')
    elif ele=='d':
        file_emotion.append('disgust')
    elif ele=='f':
        file_emotion.append('fear')
    elif ele=='h':
        file_emotion.append('happy')
    elif ele=='n':
        file_emotion.append('neutral')
    elif ele=='sa':
        file_emotion.append('sad')
    else:
        file_emotion.append('surprise')

# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Savee_df = pd.concat([emotion_df, path_df], axis=1)



# Printing the total number of samples in each dataset
print("Total number of samples in RAVDESS dataset:", len(RAVDESS_df))
print("Total number of samples in CREMA-D dataset:", len(Crema_df))
print("Total number of samples in TESS dataset:", len(Tess_df))
print("Total number of samples in SAVEE dataset:", len(Savee_df))
print("Total number of samples in all datasets: ",len(RAVDESS_df) + len(Crema_df)  + len(Tess_df) + len(Savee_df))

# combine 4 dataframes to 1.
# Savee_df, Tess_df, Crema_df, RAVDESS_df
data_path = pd.concat([Crema_df,RAVDESS_df,Tess_df,Savee_df], axis = 0)
data_path.to_csv("data_path.csv",index=False)
data_path.head()

# EMOTIONS TO TAKE
valid_emotions = ['happy', 'angry', 'sad', 'fear']

# Filter RAVDESS dataframe
RAVDESS_df_filtered = RAVDESS_df[RAVDESS_df['Emotions'].isin(valid_emotions)]

# Filter Crema dataframe
Crema_df_filtered = Crema_df[Crema_df['Emotions'].isin(valid_emotions)]

# Filter TESS dataframe
Tess_df_filtered = Tess_df[Tess_df['Emotions'].isin(valid_emotions)]

# Filter Savee dataframe
Savee_df_filtered = Savee_df[Savee_df['Emotions'].isin(valid_emotions)]

# Combine filtered dataframes
data_path_filtered = pd.concat([RAVDESS_df_filtered, Tess_df_filtered, Crema_df_filtered], axis=0)

data_path_filtered.to_csv("data_path_filtered.csv",index=False)

#extract features
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

# Extract features
FEATURES, LABELS = [], []
for path, emotion in zip(data_path_filtered.Path, data_path_filtered.Emotions):
    feature = get_features(path)
    FEATURES.append(feature)
    LABELS.append(emotion)
FEATURES = np.array(FEATURES)
print(FEATURES)
LABELS = np.array(LABELS)
encoder = OneHotEncoder()
LABELS = encoder.fit_transform(LABELS.reshape(-1, 1)).toarray()


scaler = StandardScaler()

FEATURES = scaler.fit_transform(FEATURES)
joblib.dump(scaler, "scaler.pkl")
# # making our data compatible to model.

FEATURES = np.expand_dims(FEATURES, axis=2)


# Build the model
from keras.layers import LeakyReLU
model = Sequential()

model.add(Conv1D(256, kernel_size=3, strides=1, padding='same',input_shape=(FEATURES.shape[1], 1)))
model.add(LeakyReLU(alpha=0.05))
model.add(MaxPooling1D(pool_size=3, strides=1, padding='same'))

model.add(Conv1D(256, kernel_size=3, strides=1, padding='same'))
model.add(LeakyReLU(alpha=0.05))
model.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))

model.add(Conv1D(128, kernel_size=3, strides=1, padding='same'))
model.add(LeakyReLU(alpha=0.05))
model.add(MaxPooling1D(pool_size=3, strides=1, padding='same'))

model.add(Conv1D(128, kernel_size=3, strides=1, padding='same'))
model.add(LeakyReLU(alpha=0.05))
model.add(MaxPooling1D(pool_size=3, strides=1, padding='same'))


model.add(Conv1D(64, kernel_size=3, strides=1, padding='same'))
model.add(LeakyReLU(alpha=0.05))
model.add(MaxPooling1D(pool_size=3, strides=1, padding='same'))
model.add(Conv1D(64, kernel_size=3, strides=1, padding='same'))
model.add(LeakyReLU(alpha=0.05))
model.add(MaxPooling1D(pool_size=3, strides=1, padding='same'))

model.add(Conv1D(32, kernel_size=3, strides=1, padding='same'))
model.add(LeakyReLU(alpha=0.05))
model.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))

model.add(Flatten())
model.add(Dense(units=32, activation='relu'))

model.add(Dense(units=4, activation='softmax'))
model.compile(optimizer='RMSProp',loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

