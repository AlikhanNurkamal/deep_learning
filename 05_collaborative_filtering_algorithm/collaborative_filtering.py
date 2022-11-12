import numpy as np
import tensorflow as tf
from tensorflow import keras

np.random.seed(0)


class CollaborativeFiltering:
    def __init__(self, num_movies, num_users, num_features=10):
        self.num_movies = num_movies
        self.num_users = num_users
        self.num_features = num_features

    def create_dataset(self):
        # Actual movie ratings dataset (like Y), where rows are movies and columns are users.
        movie_ratings = np.random.randint(0, 6, size=(self.num_movies, self.num_users))

        # X stands for the matrix of features of each movie. There are 'num_features' features of each movie.
        X = np.random.random(size=(self.num_movies, self.num_features))

        # W stands for the matrix of weights of each user j with 'num_features' features. Initialize the matrix with zeros.
        W = np.zeros((self.num_users, self.num_features))

        # Along with weights, there is also a bias number associated with each user.
        b = np.zeros((1, self.num_users))

        # R is a matrix that specifies whether a user j rated a movie i. If so, the value of R is 1 and 0 otherwise.
        R = np.zeros((self.num_movies, self.num_users), dtype='int8')
        for i in range(self.num_movies):
            for j in range(self.num_users):
                if movie_ratings[i, j] > 0:
                    R[i, j] = 1

        return X, W, b, movie_ratings, R

    def mean_normalization(self, Y):
        '''
        The function to apply the row mean normalization on the movie ratings.
        :param Y: original matrix of ratings. It is of size (num_movies, num_users).
        :return: Y_norm, y_mean - normalized matrix and a vector of row mean values, respectively.
        '''
        num_m, num_u = Y.shape
        # Y_norm = np.zeros((num_m, num_u))
        y_mean = np.zeros(num_m)

        for i in range(num_m):
            y_i = Y[i, :]
            y_mean[i] = np.mean(y_i)

        Y_norm = Y - y_mean.reshape(-1, 1)

        return Y_norm, y_mean

    def compute_cost(self, X, W, b, Y, R, lambda_=1.):
        '''
        The function that computes the cost function value with given parameters.
        :param X: matrix of features of each movie. It is of size (num_movies, num_features).
        :param W: matrix of weights associated with each user. It is of size (num_users, num_features).
        :param b: matrix of bias values for each user. It is of size (1, num_users).
        :param Y: original matrix of ratings. It is of size (num_movies, num_users).
        :param R: Matrix that specifies whether a movie i was rated by a user j. It is filled with 0s and 1s,
        where 1 imply that the movie was rated. It is of size (num_movies, num_users).
        :param lambda_: regularization parameter.
        :return: value of the cost function.
        '''
        J = 0   # J stands for the cost function.
        num_m, num_u = Y.shape
        for j in range(num_u):
            w_j = W[j, :]
            b_j = b[0, j]
            for i in range(num_m):
                x_i = X[i, :]
                y_ij = Y[i, j]
                r_ij = R[i, j]
                J += r_ij * np.square((np.dot(w_j, x_i) + b_j - y_ij))

        J /= 2
        J += (lambda_ / 2) * np.sum(np.square(W)) + np.sum(np.square(X))
        return J

    def compute_cost_vectorized(self, X, W, b, Y, R, lambda_):
        '''
        The function that computes the cost function value with given parameters using vectorized values.
        :param X: matrix of features of each movie. It is of size (num_movies, num_features).
        :param W: matrix of weights associated with each user. It is of size (num_users, num_features).
        :param b: matrix of bias values for each user. It is of size (1, num_users).
        :param Y: original matrix of ratings. It is of size (num_movies, num_users).
        :param R: Matrix that specifies whether a movie i was rated by a user j. It is filled with 0s and 1s,
        where 1 imply that the movie was rated. It is of size (num_movies, num_users).
        :param lambda_: regularization parameter.
        :return: value of the cost function.
        '''
        j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y) * R
        J = 0.5 * tf.reduce_sum(j ** 2) + (lambda_ / 2) * (tf.reduce_sum(X ** 2) + tf.reduce_sum(W ** 2))
        return J

    def train(self, X, W, b, Y, R, lambda_=1., iterations=201):
        '''
        The function that minimizes the cost function using the gradient descent algorithm.
        :param X: matrix of features of each movie. It is of size (num_movies, num_features).
        :param W: matrix of weights associated with each user. It is of size (num_users, num_features).
        :param b: matrix of bias values for each user. It is of size (1, num_users).
        :param Y: original matrix of ratings. It is of size (num_movies, num_users).
        :param R: Matrix that specifies whether a movie i was rated by a user j. It is filled with 0s and 1s,
        where 1 imply that the movie was rated. It is of size (num_movies, num_users).
        :param lambda_: regularization parameter.
        :param iterations: number of iterations of gradient descent algorithm.
        '''
        W = tf.Variable(W, dtype=tf.float64)
        X = tf.Variable(X, dtype=tf.float64)
        b = tf.Variable(b, dtype=tf.float64)
        optimizer = keras.optimizers.Adam(learning_rate=0.1)

        for iter in range(iterations):
            with tf.GradientTape() as tape:
                cost = self.compute_cost_vectorized(X, W, b, Y, R, lambda_)

            gradients = tape.gradient(cost, [X, W, b])
            optimizer.apply_gradients(zip(gradients, [X, W, b]))

            if iter % 10 == 0:
                print(f"Training loss at iteration {iter}: {cost}")


def main():
    model = CollaborativeFiltering(num_movies=20, num_users=100)
    X, W, b, Y, R = model.create_dataset()

    # print("X size:", X.shape)
    # print("W size:", W.shape)
    # print("b size:", b.shape)
    # print("Y size:", Y.shape)
    # print("R size:", R.shape)

    Y_norm, y_mean = model.mean_normalization(Y)
    # print("Y:", Y)
    # print("y_mean:", y_mean)
    # print("Y_norm:", Y_norm)

    cost = model.compute_cost(X, W, b, Y_norm, R, lambda_=1.5)
    print(cost)

    model.train(X, W, b, Y_norm, R, lambda_=1.5)


if __name__ == '__main__':
    main()