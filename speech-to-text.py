import os
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
import librosa
import matplotlib.pyplot as plt
import tensorflow_addons as tfa

# replace local data path relative to script
data_path ='./digits'
data_dir_list = os.listdir(data_path)
	
num_channel=1
num_classes = 10
audio_data_list=[]
INIT_LR = 1e-4
MAX_LR = 1e-2
size_data=[]

for dataset in data_dir_list:
	audio_list=os.listdir(data_path+'/'+ dataset)
    
	for audio in audio_list:
		X, sample_rate = librosa.load(data_path+'/'+dataset+'/'+audio, res_type='kaiser_fast')
		mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate,n_mfcc=60).T,axis=0) 
		audio_data_list.append(mfccs)
	size_data.append(len(audio_list))

audio_data = np.array(audio_data_list)
audio_data = audio_data.astype('float32')
audio_data /= 255

num_of_samples = audio_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

n=j=0

for i in data_dir_list:
	labels[j:j+size_data[n]]=int(i)
	j=j+size_data[n]
	n=n+1

names = []
for i in data_dir_list:
	names.append(i)
	
Y = np_utils.to_categorical(labels, num_classes)
x,y = shuffle(audio_data,Y, random_state=5)

# Split the dataset
X_train, X_rem, y_train, y_rem = train_test_split(x, y, test_size=0.2, random_state=2)
X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)

shape=X_train[0].shape

model = Sequential()
model.add(Dense(35,  activation='relu',input_shape=shape))
model.add(Dense(35,  activation='relu'))
model.add(Dense(35,  activation='relu'))
model.add(Flatten())
model.add(Dense(num_classes,activation='softmax',name='op'))

steps_per_epoch = len(X_train) #batch-size
clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=INIT_LR,
    maximal_learning_rate=MAX_LR,
    scale_fn=lambda x: 1/(2.**(x-1)),
    step_size=2 * steps_per_epoch
)

adam=keras.optimizers.Adam(learning_rate=clr,beta_1=0.9,beta_2=0.999,epsilon=None,decay=0.0,amsgrad=False)

model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=['accuracy'])

filepath=""
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history=model.fit(X_train, y_train,epochs=200, batch_size=32,callbacks=callbacks_list,validation_data=(X_valid, y_valid))

print(model.summary())

score = model.evaluate(X_test, y_test, verbose=0)

print('Test Loss:', score[0])
print('Test accuracy:', score[1])

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()