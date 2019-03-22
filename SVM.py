#still on progres#still on progress
import numpy as np
import sys

class SVM:
    def __init__(self):
        self.data = None
        self.weight = None

    def fit(self, data, label):
        self.data = data
        self.label = label
        optimization_dict = {}
        transforms = [[1,1],[1,-1],[-1,1],[-1,-1]]

        self.max_range = -1
        self.min_range = sys.maxsize
        for i in range(self.data.shape[0]):
            temp_series = self.data[i]
            #print(temp_series.max(), temp_series.min())
            self.max_range = max(self.max_range, temp_series.max())
            self.min_range = min(self.min_range, temp_series.min())

        print(self.max_range, self.min_range)

        step_sizes = [self.max_range * 0.1, self.max_range * 0.01, self.max_range * 0.001]

        b_range_multiple = 2
        b_multiple = 5
        latest_optimum = self.max_range * 10

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            optimized = False
            while not optimized:
                for b in np.arange(-1 * (self.max_range * b_range_multiple), self.max_range * b_range_multiple, step * b_multiple):
                    for transformation in transforms:
                        print(w, transformation)
                        w_t = transformation*w
                        found_option = True
                        print(w_t)
                        for i in range(self.data.shape[0]):
                            #print(np.array([w_t]).T @ np.array(self.data[i]))
                            print(self.label[i] * np.dot([w_t], self.data[i]) + b)
                            if not (self.label[i] * np.dot([w_t], self.data[i]) + b) >= 1:
                                found_option = False
                                break
                        if found_option:
                            optimization_dict[np.linalg.norm(w_t)] = [w_t, b]
                if w[0] < 0:
                    optimized = True
                else :
                    w = w - step
            norms = sorted([n for n in optimization_dict])
            optimal_choice = optimization_dict[norms[0]]
            self.w = optimal_choice[0]
            self.b = optimal_choice[1]
            latest_optimum = optimal_choice[0][0] + step*2

    def predict(self, data):
        label = np.sign(data @ self.weight + b)


