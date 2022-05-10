import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from os import getcwd
import csv
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

#Initializing path variables
path_sign_mnist_train = f"{getcwd()}/Desktop/krish/iSpeak/sign_mnist_train.csv"
path_sign_mnist_test = f"{getcwd()}/Desktop/krish/iSpeak/sign_mnist_test.csv"

#read csv using pandas
train = pd.read_csv(path_sign_mnist_train)
test= pd.read_csv(path_sign_mnist_test)

#split x (image) and y (label)
train_Y = train['label']
test_Y = test['label']
train_X = train.drop(['label'],axis = 1)
test_X = test.drop(['label'],axis = 1)

#reshape the 784 pixels to form a 28x28 image
train_X = train_X.values.reshape(27455,28,28)
test_X = test_X.values.reshape(7172,28,28)

#expand dimensions of the train_X and test_X before passing to the model
train_X = np.expand_dims(train_X,3)
test_X = np.expand_dims(test_X,3)

#callback function - called when Learning Rate Plateaus
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)

#rescaling the training and validation data before feeding it to the model
train_datagen=ImageDataGenerator(rescale=1./255)
validation_datagen=ImageDataGenerator(rescale=1./255)                             

#model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(75 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D((2,2) , strides = 2 , padding = 'same'),
    tf.keras.layers.Conv2D(50 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D((2,2) , strides = 2 , padding = 'same'),
    tf.keras.layers.Conv2D(25 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D((2,2) , strides = 2 , padding = 'same'),
    tf.keras.layers.Flatten(),    
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(25, activation='softmax')
])

# Compile the Model
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# Train the Model
history = model.fit(train_datagen.flow(train_X,train_Y,batch_size=128),
                    epochs=3 0,                    
                    validation_data=validation_datagen.flow(test_X,test_Y,batch_size=32),
                    callbacks = [learning_rate_reduction])

#save the model
model.save("iSpeakCNNModel.h5")

#plot the graphs to observe fluctuations in accuracy and loss in the training and validation set
epochs = [i for i in range(30)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
fig.set_size_inches(16,9)

ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
ax[0].plot(epochs , val_acc , 'ro-' , label = 'Testing Accuracy')
ax[0].set_title('Training & Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
ax[1].plot(epochs , val_loss , 'r-o' , label = 'Testing Loss')
ax[1].set_title('Testing Accuracy & Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
plt.show()
