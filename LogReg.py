import numpy as np
import random

class LogReg:
    def __init__(self):
        self.theta = None
        self.num_epoch = 100
        self.learning_rate = 0.001

    def sigmoid(self, e):
        return 1.0 / (1 + np.exp(-e))

    def predict(self, x):
        y_hat = self.sigmoid(np.dot(x, self.theta))
        if y_hat < 0.5:
            return -1
        else :
            return 1

    def train(self, x, y):
        y_hat = self.predict(x)
        self.theta = self.update_theta(x, y_hat, y)
        return y_hat
    def update_theta(self, x, y_hat, y) :
        return self.theta + self.learning_rate * (y - y_hat) * x

    def initiate_theta(self, mode, size):
        self.theta = np.zeros(size)
        if mode == 'random':
            for i in range(0, size):
                self.theta[i] = random.randint(-100,100)
        print("Initial theta = %s" % (self.theta))

    #initial_theta options : default = all 0
    #                      : random = randomize initial theta
    def fit(self, X, Y, num_epoch=None, learning_rate = None, initial_theta = 'default'):
        if num_epoch is not None:
            self.num_epoch = num_epoch
        if learning_rate is not None :
            self.learning_rate = learning_rate

        self.initiate_theta(initial_theta, X.shape[1])

        min_accuracy = 100
        max_accuracy = 0
        for i in range(0, self.num_epoch):
            false_count = 0
            print("Running with mode %s with learning rate %f. Number of epoch %d" % (initial_theta, self.learning_rate, self.num_epoch))
            for j in range(0, len(Y)):
                y_hat = self.train(X[j], Y[j])
                if y_hat != Y[j]:
                    false_count = false_count + 1
            accuracy = (1 - (false_count/len(Y))) * 100
            min_accuracy = min(accuracy, min_accuracy)
            max_accuracy = max(accuracy, max_accuracy)
            print("Epoch : %d. Current theta : %s. Accuracy : %.3f" % (i+1, self.theta, accuracy))
        print("Max accuracy = %.3f. Min accuracy = %.3f" % (max_accuracy, min_accuracy))

    def test(self, X, Y):
        false_count = 0
        for i in range(0, len(Y)):
            y_hat = self.predict(X[i])
            if y_hat != Y[i]:
                false_count = false_count + 1
        accuracy = (1 - false_count / len(Y)) * 100
        print("Test result :")
        print("Test data count : %d" % (len(Y)))
        print("Accuracy : %.3f" % (accuracy))




