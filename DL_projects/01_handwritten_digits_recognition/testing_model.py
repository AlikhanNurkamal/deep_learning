import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

mnist = keras.datasets.mnist
( X_train, y_train ), ( X_test, y_test ) = mnist.load_data( )
X_test = X_test.reshape( len( X_test ), 28 * 28 )
X_test = X_test.astype( "float32" )
X_test /= 255

model = load_model( "model.h5" )

logits = model( X_test )
predictions = np.array( [np.argmax( i ) for i in tf.nn.softmax( logits )] )
# print( predictions[:50] )
# print( y_test[:50] )

result = np.concatenate( ( predictions.reshape( -1, 1 ), y_test.reshape( -1, 1 ) ), axis = 1 )
np.savetxt( fname = "preds_true.txt", X = result, fmt = "%d",
            header = "Predicted  True", delimiter = '\t\t' )
