import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist

( X_train, y_train ), ( X_test, y_test ) = mnist.load_data( )

# Reshaping the input into len( ) x 784, because the images are 28 pixels by 28 pixels.
X_test = X_test.reshape( len( X_test ), 28 * 28 )
X_test = X_test.astype( "float32" )
X_test /= 255

model = keras.models.load_model( "model.h5" )
logits = model( X_test )
prediction = np.array( [np.argmax( i ) for i in tf.nn.softmax( logits )] )

result = np.concatenate( ( prediction.reshape( -1, 1 ), y_test.reshape( -1, 1 ) ), axis = 1 )
np.savetxt( "preds_true.txt", result, fmt = "%d", header = "Prediction  True", delimiter = "\t\t" )
