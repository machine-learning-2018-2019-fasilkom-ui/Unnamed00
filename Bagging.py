import numpy as np
from LogReg import LogReg
import random

class Bagging:
    def __init__(self):
        self.seed = 585
        self.group_size = 10
        self.train_samples = List()
        self.train_labels = List()
        self.models = List()
        self.logreg = LogReg()

    def generate_samples(self, X, Y, group_size = None, seed = None):
        if seed is not None :
            self.seed = seed

        if group_size is not None:
            self.group_size = group_size

        seed(self.seed)
        for i in range(0, self.group_size):
            self.randomize_sample(X,Y)


    def randomize_sample(self, X, Y):
        indices = random.randint(0, len(Y), len(Y))
        temp_sample = np.take(X, indices)
        print(temp_sample.shape())
        temp_label = np.take(X, indices)
        print(temp_label.shape())

        self.train_samples.append(temp_sample)
        self.train_labels.append(temp_label)

    def fit(self):
        for i in range(0,self.group_size):
            self.models.append(self.logreg.fit(self.train_samples(i), self.train_labels(i)))

    #add test method for easier testing
    def test(self, X, Y):
        test_size = len(Y)
        for i in range(0, test_size):
            class_dict =
            for model in self.models:
                y_hat = model.predict(X[i])



