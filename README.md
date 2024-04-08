
# Voice Recognition

Imagine a conference room, a meeting is taking place. There are several people, they all take turns to speak. We want to transcribe what they are saying. There are tools that can transcribe voice to text (most of us have them on our phones already), but can we train a model to learn voices and accurately predict who is speaking just by listening to their voice? That is all this project is all about.



## Contents

- Problem Statement
- Introduction
- Summary of the Project
- Results
- Future Explorations
- Sources and Dataset



## Problem Statement

This project is based on a voice recognition model. Our primary goal would be to recognize the person speaking the words rather than the words spoken. Speaker identification would determine if a registered speaker provides a given utterance and Speaker verification would declare the identity claim of the speaker.
## Introduction

- Our goal is to develop a model that can effectively and accurately identify a registered speaker.There are two primary methods which we will employ in our model development process and then we will use a voting classifier to combine the predictions of these two methods. 
- The first method involves using the python library librosa to extract numerical features from the audio clips, which will be utilized to train a neural network (NN) model. 
- The second approach involves converting the audio clips into images, which will then be used to train a convolutional neural network (CNN) model.

## Summary of the Project

We had a dataset consisting of voice folders of 33 speakers i.e. 33 classes. Each speaker folder has audio samples of the speakers.
Then, the first problem was to learn how to manipulate audio data and build models to classify sounds. We worked upon two neural network models throughout this project in order to get good accuracy.

- FEED FORWARD NEURAL NETWORK

Librosa is a fantastic library for python to use with audio files and it is what most people used in the audio classification problems. We used the librosa library to extract features.Also we used the following libraries- 
- Tensorflow
- Keras
- Pandas
- Numpy
- Librosa
- Matplotlib
- Scikit-Learn

Further We installed resampy as a requirement for Efficient sample rate conversion in Python.  We extracted the features through a function that iterates through every row of the dataframe accessing the file in the computer by reading the file's path. We used Mel-frequency Cepstral Coefficients (MFCCs), Chromagram, Spectral Contrast and Tonal Centroid Features (tonnetz). The extracted features are then normalized and stored in a list and the subdirectory containing the file is stored in another list. we got an array of numerous features which we stored in a dataframe and we also did Hot label encoding creating a binary vector for each unique value in the categorical data.
Moving further, we used the train_test_split function from scikit-learn to split a dataset into training, validation, and test sets. 
We then trained a Keras based Feed Forward neural network.
We then trained a neural network model with 300 epochs and got 90.52% training accuracy and 87.26% validation accuracy.


- CONVOLUTIONAL NEURAL NETWORK

We used the same dataset as used in the previous model stored in a folder named "finalaudio". Then we created a function. For each file in the input list, the function loads the audio data using the librosa library.The function calculates the mel spectrogram of the audio data using librosa.feature.melspectrogram.The function saves the resulting image as a JPEG file.

We created a CSV by appending the filenames in it and creating a column named as ‘audio_filenames’.
Then converted it into a dataframe with 1744 rows .
We defined a function named ‘make_label’ which is responsible for splitting the file names and defined a function named ‘make_jpg’ through which we could change the names of the files from ‘.flac’ to ‘.flac.jpg’.
Further we loaded a set of images from a directory, resizes them to a target size of (128,128), and converts them to NumPy arrays using Keras' image preprocessing module. Then we split the input data X and target data y into three sets: training set, validation set, and test set and converted a 1-dimensional array of integer labels y into a one-hot encoded format that can be used as target data.
We trained using normal CNN model with 100 epochs and with a batch size of 8 , we got 93.84% training accuracy and 97.45% validation accuracy.




## Results

- For FFNN, We trained the neural network model with 300 epochs and got 90.52% training accuracy and 87.26% validation accuracy.
- For CNN, We trained model with 100 epochs and with a batch size of 8 getting a great 93.84% training accuracy and 97.45% validation accuracy.

## Future explorations

- We would like to gridsearch over the best parameters on our CNN and see if we could get the same or better accuracy than our Dense Layered Neural Network.
- We would also like to fit a Recurrent Neural Network and Inception V3(although we tried this but faced the problem of excessive RAM usage) and see how accurate it is since they are good with time series data and voice clips are basically time series.
- We would like to try CNN models with all the different kinds of images that librosa provides from audio files and see which kind of images give better predictions.

- After working on this, we can easily try to work on the gender classification model in the future.
## Sources and Dataset

- Dataaset for the NN model:-

https://drive.google.com/drive/folders/1yqE_-f1p93o0sVntdXpE3iw0BPHLkty6?usp=share_link

- Dataset for the CNN model:-

https://drive.google.com/drive/folders/1fTlgHdwP5lwMEmSED3qoHtEvuD-FWgP6?usp=share_link


- Sources

- http://www.openslr.org/12/
- https://medium.com/@patrickbfuller/librosa-a-python-audio-libary-60014eeaccfb
- https://librosa.github.io/librosa/0.6.0/feature.html
- https://medium.com/@CVxTz/audio-classification-a-convolutional-neural-network-approach-b0a4fce8f6c
- https://www.analyticsvidhya.com/blog/2017/08/audio-voice-processing-deep-learning/
- https://www.endpoint.com/blog/2019/01/08/speech-recognition-with-tensorflow
- https://keras.io/preprocessing/image/
- https://keras.io/models/sequential/