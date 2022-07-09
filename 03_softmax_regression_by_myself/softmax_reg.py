import numpy as np


def my_softmax( a_in, W, b ):
    '''
    :param a_in: output vector of the previous layer a[l-1] that is an input for current layer.
    :param W: matrix of weights of different units of current layer.
    :param b: vector of biases of different units of current layer.
    :return: a_out - vector of probabilities of each predicting class.
    '''
    units = W.shape[1]
    z = np.zeros( units )
    a_out = np.zeros( units )
    for i in range( units ):
        w_i = W[:, i]
        z[i] = np.dot( a_in, w_i ) + b[i]

    for i in range( units ):
        e_z = np.exp( z )
        a_out[i] = e_z[i] / np.sum( e_z )

    return a_out


def my_sequential( input, W1, b1, W2, b2, W3, b3 ):
    a_1 = my_softmax( input, W1, b1 )
    a_2 = my_softmax( a_1, W2, b2 )
    a_3 = my_softmax( a_2, W3, b3 )
    return a_3


np.random.seed( 10 )

weights = np.array( np.random.randn( 10, 10 ) )
X = np.array( np.random.randn( 10, ) )
biases = np.array( np.random.randn( 10, ) )

print( my_softmax( X, weights, biases ) )
# According to these weights and input vector, the single softmax layer predicts that the
# probability of the result being of class 3 is 50%, so the algorithm predicts it is a class 3.
# P.S. these numbers are random for the sake of testing the algorithm.


W_2 = np.array( np.random.randn( 10, 5 ) )
b_2 = np.array( np.random.randn( 5, ) )
W_3 = np.array( np.random.randn( 5, 3 ) )
b_3 = np.array( np.random.randn( 3, ) )

print( my_sequential( X, weights, biases, W_2, b_2, W_3, b_3 ) )
# 3 layers of softmax activation function predict that the result is of class 1
# with the output probability of almost 73%.
