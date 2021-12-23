# speech-to-text
Python script to recognize the spoken digits 0-9 using an ANN.

#Approach/Algorithm:
Librosa is a well known python library for sound analysis. Using librosa, we can load the audio file into a floating point time series that can be understood by the machine.
Librosa has additional functions to reduce the sample load time.
MFCC (Mel-frequency cepstrum) is the most popular feature extractor in speech recognition. Feature extraction is the transformation of input data to features. The resultant feature vector will be fed into an artificial neural network with hidden layers. Using the librosa mfcc function, we can extract the desired number of features into a vector. This is done iteratively for each audio file in the dataset. Finally, the vectors are normalized and the corresponding labels are created. 
The dataset is then split into training, validation and testing sets. The next step is feeding the training and validation data into a sequential neural network. Finally, the trained model is evaluated against the test set and the score is computed.

![alt text](https://github.com/nirmitshetty/speech-to-text/blob/main/approach.png?raw=true)
