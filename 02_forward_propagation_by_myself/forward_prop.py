import numpy as np


def sigmoid( z ):
    return 1 / ( 1 + np.exp( -z ) )


def dense( a_in, W, b ):
    '''
    :param a_in: the activation vector of the previous layer
    :param W: a matrix comprised of vectors w:
    if w1_1 = [1, 2], w1_2 = [-3, 4], and w1_3 = [5, -6], then
    W = [[1, -3, 5],
         [2, 4, -6]]
    :param b: an array of parameters b
    :return: activation vector of current layer
    '''

    number_of_units = W.shape[1]
    a_out = np.zeros( number_of_units )
    for i in range( number_of_units ):
        w = W[:, i]
        a_out[i] = sigmoid( np.dot( w, a_in ) + b[i] )

    return a_out


def sequential( x ):
    # a1 = dense( x, W1, b1 )
    # a2 = dense( a1, W2, b2 )
    # a3 = dense( a2, W3, b3 )
    # a4 = dense( a3, W4, b4 )
    f_x = 0
    # f_x = a4
    return f_x


x = np.array( [200., 17.] )
weights = np.array( [[1, -3, 5],
                     [2, 4, -6]] )
params = np.array( [-1, 1, 2] )

print( dense( x, weights, params ) )
# print( sequential( x ) )