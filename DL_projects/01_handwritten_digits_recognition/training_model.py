import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.losses import SparseCategoricalCrossentropy

mnist = keras.datasets.mnist
( X_train, y_train ), ( X_test, y_test ) = mnist.load_data( )

# Reshaping the input values to len( ) x 784, as images are 28 pixels by 28 pixels.
X_train = X_train.reshape( len( X_train ), 28 * 28 )
X_test = X_test.reshape( len( X_test ), 28 * 28 )
X_train = X_train.astype( "float32" )
X_test = X_test.astype( "float32" )

# Normalizing inputs to be within the range [0, 1]
X_train /= 255
X_test /= 255

model = tf.keras.models.Sequential( [
    keras.layers.Dense( units = 512, activation = "relu" ),
    keras.layers.Dense( units = 256, activation = "relu" ),
    keras.layers.Dense( units = 128, activation = "relu" ),
    keras.layers.Dense( units = 10, activation = "linear" )
] )

# print( model.summary( ) )

model.compile( optimizer = "adam",
               loss = SparseCategoricalCrossentropy( from_logits = True ),
               metrics = ["accuracy"] )
model.fit( X_train, y_train, batch_size = 128, epochs = 10, validation_split = 0.2 )
test_loss, test_acc = model.evaluate( X_test, y_test )
print( "Test accuracy:", test_acc )

model.save( "model.h5" )
print( "Model saved!" )
