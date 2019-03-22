import numpy as np

class LogReg:
    def __init__:
        self.weights = None

    def sigmoid(e):
        return 1.0 / (1 + np.exp(-e))

    def predict(X, weights):
        return sigmoid(np.dot(X, self.weights))

    def train(features, y, weights, learning_rate):
        N = len(features)
        y_apostrophe = predict(features, weights)
        gradient = np.dot(features.T,  y_apostrophe - y)
        self.weights -= (gradient / N) * learning_rate

