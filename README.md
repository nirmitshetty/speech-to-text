# speech-to-text
Python script to recognize the spoken digits 0-9 using an ANN.

# Available Datasets
Listed below are two open speech dataset consisting of audio recordings in wav file format. The audio has already been preprocessed by trimming silence at the start and end.

- [AudioMNIST](https://github.com/soerenab/AudioMNIST)
- [Free spoken digit dataset](https://github.com/Jakobovski/free-spoken-digit-dataset)

*Note- The script expects the audio files to be pre segregated in their corresponding labelled folders.*
E.g:  
/digits  
/digits/0  
/digits/1  
/digits/2  
.  
.  
.  

# Approach/Algorithm:
Librosa is a well known python library for sound analysis. Using librosa, we can load the audio file into a floating point time series that can be understood by the machine.
Librosa has additional functions to reduce the sample load time.
MFCC (Mel-frequency cepstrum) is the most popular feature extractor in speech recognition. Feature extraction is the transformation of input data to features. The resultant feature vector will be fed into an artificial neural network with hidden layers. Using the librosa mfcc function, we can extract the desired number of features into a vector. This is done iteratively for each audio file in the dataset. Finally, the vectors are normalized and the corresponding labels are created. 
The dataset is then split into training, validation and testing sets. The next step is feeding the training and validation data into a sequential neural network. Finally, the trained model is evaluated against the test set and the score is computed.
Below is the summary of the process described above.

![alt text](https://github.com/nirmitshetty/speech-to-text/blob/main/approach.png?raw=true)

# Model metrics
- MFCC sampling rate: default value is 20.
- It’s important to use separate data for training and evaluation to get unbiased results. Validation set is for criticizing whether our training is moving in the right
direction or not. There is no optimal split. Every model has its unique needs for a split percentage. However, there is a rough standard that we can follow. We went with the 80:10:10 split ratio for our training, validation and test datasets.
- Hidden layers and number of neurons:There is no single way for determining a good network topology. Just keep adding layers until the test error does not improve anymore.
- Activation functions:ReLU & Softmax. Softmax is the preferred choice for activation function for multiclass classification where the classes are mutually exclusive.
- Dropout Regularization:We did not observe overfitting.
- Otimizer & Learning Rate: Adam with cyclic learning rate
- Loss:Cross-entropy is the default loss function used for multi-class classification problems.
- Evaluation: Accuracy is a good performance measure for fairly balanced datasets.

# Results
Our model gave 90%+ test accuracy for both datasets.
