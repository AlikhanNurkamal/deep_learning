import numpy as np
import matplotlib.pyplot as plt


def calculate_mu_and_delta( X ):
    m, n = X.shape
    mu = np.array( np.zeros( n ) )
    delta = np.array( np.zeros( n ) )

    for j in range( n ):
        for i in range( m ):
            mu[j] += X[i, j]

        mu[j] /= m

        for i in range( m ):
            delta[j] += ( X[i, j] - mu[j] ) ** 2

        delta[j] /= m

    return mu, delta


def calculate_probability( mu, delta, new_x ):
    n = len( new_x )
    p = 1

    for j in range( n ):
        p *= ( 1 / ( np.sqrt( 2 * np.pi ) * delta[j] ) * np.exp( -( new_x[j] - mu[j] ) ** 2 / ( 2 * delta[j] ** 2 ) ) )

    return p


def main( ):
    dataset = np.array( np.zeros( ( 500, 3 ) ) )

    dataset[:, 0] = np.random.normal( 11.54, 1, 500 )
    dataset[:, 1] = np.random.normal( 52.2, 1.22, 500 )
    dataset[:, 2] = np.random.normal( 108.46, 1.45, 500 )

    new_point = np.array( [11.12, 54.1, 120.5] )

    mu, delta = calculate_mu_and_delta( dataset )
    print( calculate_probability( mu, delta, new_point ) )

    fig = plt.figure( )
    ax = fig.add_subplot( 111, projection='3d' )
    ax.scatter3D( dataset[:, 0], dataset[:, 1], dataset[:, 2], marker='x' )
    ax.scatter3D( new_point[0], new_point[1], new_point[2], c='r', marker='x' )

    ax.set_xlabel( "feature x0" )
    ax.set_ylabel( "feature x1" )
    ax.set_zlabel( "feature x2" )

    plt.show( )


if __name__ == '__main__':
    main( )
