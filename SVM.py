import numpy as np
import cvxopt
import math
from kernels import kernels

class SVM:
    def __init__(self, kernel=kernels.linear, C=None, degree=3, sigma=5, gamma=2):
        self.a = None
        self.sv = None
        self.svt = None
        self.b = None
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.sigma = sigma
        self.gamma = gamma
        if(self.C is not None):
            self.C = float(self.C)

    def _qpSolver(self, X, y):
        n, features = X.shape

        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i,j] = self._kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n) * -1)
        A = cvxopt.matrix(y, (1,n), 'd')
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n) * -1))
            h = cvxopt.matrix(np.zeros(n))
        else:
            G = cvxopt.matrix(np.vstack((np.diag(np.ones(n) * -1), np.identity(n))))
            h = cvxopt.matrix(np.hstack((np.zeros(n), np.ones(n) * self.C)))

        cvxopt.solvers.options['show_progress'] = False
        res = cvxopt.solvers.qp(P, q, G, h, A, b)
        a = np.ravel(res['x'])
        return a

    def _kernel(self, x, y):
        if self.kernel == kernels.linear:
            return kernels.linear(x, y)
        elif self.kernel == kernels.polynomial:
            return kernels.polynomial(x, y, self.degree)
        elif self.kernel == kernels.rbf:
            return kernels.rbf(x, y, self.sigma)

        raise Exception('Unrecognized kernel option')

    def find_sv(self, X,y):
        a = self._qpSolver(X,y)
        sv = []
        svt = []
        a_sv = []
        for i in range(0, len(y)):
            if not math.isclose(a[i],0):
                sv.append(X[i])
                svt.append(y[i])
                a_sv.append(a[i])

        self.sv = np.array(sv)
        self.svt = np.array(svt)
        self.a = np.array(a_sv)

    def calculate_intercept(self, X, y):
        b = 0
        for i in range(0, len(self.a)):
            b += self.svt[i]
            b -= np.sum(self.a * self.svt[i] * self._kernel(self.sv[i], self.sv[i]))
        return b/len(self.a)

    def calculate_weight(self):
        weight = 0
        for i in range(0, len(self.svt)):
            weight += self.a[i] * self.svt[i] * self.sv[i]

        return np.array(weight)

    def fit(self, X, y):
        self.find_sv(X,y)
        self.b = self.calculate_intercept(X,y)

        true_count = 0
        y_hat, y_val = self.predict(X)
        for i in range(0, len(y)):
            if y_hat[i] == y[i]:
                true_count += 1

        print("Accuracy : %.3f" % ((true_count/len(y))*100))

    def predict(self, X):
        y_predict = None
        weight = None
        if self.kernel == kernels.linear:
            weight = self.calculate_weight()

        y_predict = np.zeros(X.shape[0])
        for i in range(0, X.shape[0]):
            if weight is not None:
                y_predict[i] = np.dot(weight.T, X[i])
            else:
                temp_predict = 0
                for j in range(0, len(self.a)):
                    temp_predict += self.a[j] * self.svt[j] * self._kernel(X[i], self.sv[j])
                y_predict[i] = temp_predict
        print(np.sign(y_predict + self.b))
        return np.sign(y_predict + self.b), y_predict + self.b

