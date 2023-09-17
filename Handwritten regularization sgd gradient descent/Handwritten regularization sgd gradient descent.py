import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from numpy.linalg import *
np.random.seed(42)  # don't change this line


class LinearRegression:
    """
    Linear Regression

    Parameters
    ----------
    alpha: float, default=0.01
        Learning rate
    tol : float, default=0.0001
        Tolerance for stopping criteria
    max_iter : int, default=10000
        Maximum number of iterations of gradient descent
    theta_init: None (or) numpy.ndarray of shape (D + 1,)
        The initial weights; if None, all weights will be zero by default
    penalty : string, default = None
        The type of regularization. The other acceptable options are l1 and l2
    lambd : float, default = 1.0
        The parameter regularisation constant (i.e. lambda)

    Attributes
    ----------
    theta_ : numpy.ndarray of shape (D + 1,)
        The value of the coefficients after gradient descent has converged
        or the number of iterations hit the maximum limit
    hist_theta_ : numpy.ndarray of shape (num_iter, D + 1) where num_iter is the number of gradient descent iterations
        Stores theta_ after every gradient descent iteration
    hist_cost_ : numpy.ndarray of shape (num_iter,) where num_iter is the number of gradient descent iterations
        Stores cost after every gradient descent iteration
    """

    def __init__(self, alpha=0.01, tol=1e-4, max_iter=100, theta_init=None, penalty=None, lambd=0):

        # store meta-data
        self.alpha = alpha
        self.theta_init = theta_init
        self.max_iter = max_iter
        self.tol = tol
        self.penalty = penalty
        self.lambd = lambd

        self.theta_ = None
        self.hist_cost_ = None
        self.hist_theta_ = None

    def compute_cost(self, theta, X, y):

        """
        Compute the cost/objective function.

        Parameters
        ----------
        theta: numpy.ndarray of shape (D + 1,)
            The coefficients
        X: numpy.ndarray of shape (N, D + 1)
            The features matrix
        y: numpy.ndarray of shape (N,)
            The target variable array

        Returns
        -------
        cost: float
            The cost as a scalar value
        """
        hxi_yi = 0
        for i in range(len(X)):
            hxi = np.multiply(theta, X[i])
            hxi_yi += (hxi[0] + hxi[1] - y[i]) ** 2

        ans = hxi_yi / (len(X))

        if self.penalty == 'l1':
            return ans + self.lambd * sum(abs(theta[1:]))
        elif self.penalty == 'l2':
            return ans + self.lambd * sum([num ** 2 for num in theta[1:]])
        else:
            return ans


    def compute_gradient(self, theta, X, y):

        """
        Compute the gradient of the cost function.

        Parameters
        ----------
        theta: numpy.ndarray of shape (D + 1,)
            The coefficients
        X: numpy.ndarray of shape (N, D + 1)
            The features matrix
        y: numpy.ndarray of shape (N,)
            The target variable array

        Returns
        -------
        gradient: numpy.ndarray of shape (D + 1,)
            The gradient values
        """
        if self.penalty == 'l2':
            gradient_vals = (2 / len(X)) * (np.dot(np.dot(np.transpose(X), X), theta) - np.dot(np.transpose(X), y))
            theta_no_intercrpt_l2 = theta
            theta_no_intercrpt_l2[0] = 0
            return gradient_vals + 2 * self.lambd * theta_no_intercrpt_l2
        elif self.penalty == 'l1':
            gradient_vals = (2 / len(X)) * (np.dot(np.dot(np.transpose(X), X), theta) - np.dot(np.transpose(X), y))
            theta_no_intercrpt_l1 = [0]
            for i in range(1, len(theta)):
                theta_no_intercrpt_l1.append(-1 * self.lambd if 0 > theta[i] else 1 * self.lambd)
            return gradient_vals + theta_no_intercrpt_l1
        else:
            gradient_vals = (2 / len(X)) * (np.dot(np.dot(np.transpose(X), X), theta) - np.dot(np.transpose(X), y))
        return gradient_vals


    def has_converged(self, theta_old, theta_new):

        """
        Return whether gradient descent has converged.

        Parameters
        ----------
        theta_old: numpy.ndarray of shape (D + 1,)
            The weights prior to the update by gradient descent
        theta_new: numpy.ndarray of shape (D + 1,)
            The weights after the update by gradient descent

        Returns
        -------
        converged: bool
            Whether gradient descent converged or not
        """

        norm_2_x = np.linalg.norm(theta_new - theta_old, ord=2)
        return self.tol >= norm_2_x


    def fit(self, X, y):

        """
        Compute the coefficients using gradient descent and store them as theta_.

        Parameters
        ----------
        X: numpy.ndarray of shape (N, D)
            The features matrix
        y: numpy.ndarray of shape (N,)
            The target variable array

        Returns
        -------
        Nothing
        """

        N, D = X.shape

        # Adding a column of ones at the beginning for the bias term
        ones_col = np.ones((N, 1))
        X = np.hstack((ones_col, X))

        # Initializing the weights
        if self.theta_init is None:
            theta_old = np.zeros((D + 1,))
        else:
            theta_old = self.theta_init

        # Initializing the historical weights matrix
        # Remember to append this matrix with the weights after every gradient descent iteration
        self.hist_theta_ = np.array([theta_old])

        # Computing the cost for the initial weights
        cost = self.compute_cost(theta_old, X, y)

        # Initializing the historical cost array
        # Remember to append this array with the cost after every gradient descent iteration
        self.hist_cost_ = np.array([cost])

        epoch = 0
        self.hist_theta_ = np.append(self.hist_theta_, np.array([self.hist_theta_[epoch] -
                                                                 self.alpha * self.compute_gradient(
            self.hist_theta_[epoch], X, y)]), axis=0)

        epoch = 1
        while (self.has_converged(self.hist_theta_[epoch], self.hist_theta_[epoch - 1]) == 0 and epoch < self.max_iter):
            self.hist_cost_ = np.append(self.hist_cost_, [self.compute_cost(self.hist_theta_[epoch], X, y)], axis=0)
            self.hist_theta_ = np.append(self.hist_theta_, [self.hist_theta_[epoch] -
                                                            self.alpha * self.compute_gradient(self.hist_theta_[epoch],
                                                                                               X, y)], axis=0)
            epoch += 1
        self.theta_ = self.hist_theta_[-1]

    def fit_sgd(self, X, y):

        """
        Compute the coefficients using gradient descent and store them as theta_.

        Parameters
        ----------
        X: numpy.ndarray of shape (N, D)
            The features matrix
        y: numpy.ndarray of shape (N,)
            The target variable array

        Returns
        -------
        Nothing
        """

        N, D = X.shape

        # Adding a column of ones at the beginning for the bias term
        ones_col = np.ones((N, 1))
        X = np.hstack((ones_col, X))

        # Initializing the weights
        if self.theta_init is None:
            theta_old = np.zeros((D + 1,))
        else:
            theta_old = self.theta_init

        # Initializing the historical weights matrix
        # Remember to append this matrix with the weights after every gradient descent iteration
        self.hist_theta_ = np.array([theta_old])

        # Computing the cost for the initial weights
        cost = self.compute_cost(theta_old, X, y)

        # Initializing the historical cost array
        # Remember to append this array with the cost after every gradient descent iteration
        self.hist_cost_ = np.array([cost])

        epoch = 0
        index = 0
        for iteration, x in enumerate(X):
            self.hist_theta_ = np.append(self.hist_theta_, np.array([self.hist_theta_[index] -
                                                                     self.alpha * self.compute_gradient(
                self.hist_theta_[index], np.array([x]), np.array([y[iteration]]))]), axis=0)
            index += 1
        epoch = 1
        while self.has_converged(self.hist_theta_[index], self.hist_theta_[index - 1]) == 0 and epoch < self.max_iter:
            for iteration, x in enumerate(X):
                self.hist_theta_ = np.append(self.hist_theta_, np.array([self.hist_theta_[index] -
                                                                         self.alpha * self.compute_gradient(
                    self.hist_theta_[index], np.array([x]), np.array([y[iteration]]))]), axis=0)
                index += 1
            epoch += 1
        self.theta_ = self.hist_theta_[-1]

    def predict(self, X):

        """
        Predict the target variable values for the data points in X.

        Parameters
        ----------
        X: numpy.ndarray of shape (N, D)
            The features matrix

        Returns
        -------
        y_hat: numpy.ndarray of shape (N,)
            The predicted target variables values for the data points in X
        """

        N = X.shape[0]
        X = np.hstack((np.ones((N, 1)), X))
        return np.dot(self.theta_, np.transpose(X))


def test_lin_reg_predict_sgd(LinearRegression):
    lr_reg = LinearRegression(max_iter=5)
    np.random.seed(1)
    test_case_X = np.random.randn(50, 2)
    test_case_y = np.random.randint(0, 2, 50)
    lr_reg.fit_sgd(test_case_X, test_case_y)
    ans = lr_reg.predict(test_case_X)
    required_ans = np.array([0.4113478, 0.28834918, 0.1227324, 0.39008601, 0.43987045, 0.17506316,
                             0.40365951, 0.32180596, 0.32776898, 0.56721846, 0.63147595, 0.57385561,
                             0.38334306, 0.31959516, 0.5517445, 0.39322627, 0.3213112, 0.45537132,
                             0.48490982, 0.62956115, 0.32575875, 0.72747134, 0.37152396, 0.81428507,
                             0.57451273, 0.42292006, 0.3905908, 0.56212164, 0.64126265, 0.62130162,
                             0.65671342, 0.43645374, 0.47163355, 0.74245718, 0.29808437, 0.35882346,
                             0.61700668, 0.15509352, 0.59866825, 0.60026664, 0.43537041, 0.5427557,
                             0.49628385, 0.51805151, 0.65681787, 0.52965323, 0.36155917, 0.49471154,
                             0.47184886, 0.57066729])

    assert np.mean(np.abs(ans - required_ans)) <= 1e-2


test_lin_reg_predict_sgd(LinearRegression)