import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist

( X_train, y_train ), ( X_test, y_test ) = mnist.load_data( )

# Reshaping the input into len( ) x 784, because the images are 28 pixels by 28 pixels.
X_train = X_train.reshape( len( X_train ), 28 * 28 )
X_test = X_test.reshape( len( X_test ), 28 * 28 )
X_train = X_train.astype( "float32" )
X_test = X_test.astype( "float32" )

# Normalizing inputs to be within the range [0, 1]
X_train /= 255
X_test /= 255

print( X_train.shape[0], "training samples." )
print( X_test.shape[0], "testing samples." )

model = tf.keras.models.Sequential( [
    tf.keras.layers.Dense( units = 512, activation = "relu" ),
    tf.keras.layers.Dense( units = 256, activation = "relu" ),
    tf.keras.layers.Dense( units = 128, activation = "relu" ),
    tf.keras.layers.Dense( units = 10 )
] )

model.compile( loss = tf.keras.losses.SparseCategoricalCrossentropy( from_logits = True ),
               optimizer = "adam",
               metrics = ["accuracy"] )
model.fit( X_train, y_train, batch_size = 128, epochs = 10, validation_split = 0.2 )

test_loss, test_acc = model.evaluate( X_test, y_test )
print( "Test accuracy:", test_acc )

model.save( "model.h5" )
print( "Model saved!" )
