import numpy as np
from numpy.linalg import norm

class kernels:
    def __init__(self):
        pass

    @staticmethod
    def linear(x, y):
        sum = 0
        for i in range(0, len(x)):
            sum += x[i] * y[i]

        return sum

    @staticmethod
    def polynomial(x, y, degree):
        return (1 + kernels.linear(x, y))**degree

    @staticmethod
    def rbf(x, y, sigma, gamma):
        e = 2.718281828459045
        list_x_y = []
        for i in range(0, len(x)):
            list_x_y.append(x[i]-y[i])

        norm_squared = norm(list_x_y)**2

        return e**(-norm_squared/(2*(sigma**2)))

