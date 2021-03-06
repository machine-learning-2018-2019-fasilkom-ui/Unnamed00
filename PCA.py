import functools as ftools
import numpy as np
import pandas as pd

class PCA:
    def __init__(self, verbose=True):
        self.data = None
        self.threshold = 0.95
        self.verbose = verbose

    def calculate_mean(self, arr):
        return ftools.reduce(lambda x,y: x+y, arr)/len(arr)

    def calculate_centered_columns_data(self, means):
        for i in range(0, len(self.data)):
            for j in range(0, len(means)):
                self.data[i][j] = self.data[i][j] - means[j]

    def calculate_covariance_matrix(self):
        return np.dot(self.data.T, self.data) / len(self.data)
        #return np.cov(arr, rowvar=0, bias=1)

    def calculate_eigen_values(self, arr):
        m,v = np.linalg.eigh(arr)
        sorted_v = v[:, m.argsort()[::-1]]
        sorted_m = m[m.argsort()[::-1]]

        return sorted_m, sorted_v

    def calculate_smallest_dimensionality(self, eigen_values, threshold):
        sum_of_eigen_values = ftools.reduce(lambda x,y: x+y, eigen_values)
        current_sum = 0;
        for i in range(0, len(eigen_values)):
            current_sum += eigen_values[i]
            if (current_sum/sum_of_eigen_values) >= threshold :
                return i+1

    def reduce_dimensions(self, data, threshold=None):
        if threshold is not None:
            self.threshold = threshold
        self.data = data.to_numpy()

        #calculate means of all columns
        means = []
        for i in range(len(self.data[0])):
            means.append(self.calculate_mean(self.data[:,i]))

        self.calculate_centered_columns_data(means)
        covmats = self.calculate_covariance_matrix()
        sorted_eigen_values, sorted_basis_vector = self.calculate_eigen_values(covmats)
        if self.verbose:
            print("Sorted eigen values :\n")
            print(sorted_eigen_values)

        #get samllest dimensionality based on eigen values and threshold
        smallest_dimensionality = self.calculate_smallest_dimensionality(sorted_eigen_values, self.threshold)

        print("Dimensionality reduction result : ")
        print("  - Final data feature size : %d" %(smallest_dimensionality))

        #calculate transformed data by multiplying centered data with basis vector
        self.data = np.dot(data, sorted_basis_vector[:,:smallest_dimensionality])
        print("  - Final data feature matrix : (%d, %d)" % (self.data.shape[0], self.data.shape[1]))
        if self.verbose :
            print(self.data)
        return pd.DataFrame(self.data)

