import numpy as np
from LogReg import LogReg

class Bagging:
    def __init__(self, models):
        self.seed = None
        self.group_size = len(models)
        self.train_samples = []
        self.train_labels = []
        self.models = models

    def generate_samples(self, X, Y, seed_val = 585):
        self.seed = seed_val

        np.random.seed(self.seed)
        for i in range(0, self.group_size):
            self.randomize_sample(X,Y)

        print("train samples : %d \ntrain labels : %d " % (len(self.train_samples), len(self.train_labels)))

    #generate randomized sample based on radint and a given seed
    def randomize_sample(self, X, Y):
        indices = np.random.randint(0, len(Y), len(Y))
        temp_sample = X[indices]
        #print(temp_sample.shape)
        temp_label = Y[indices]
        #print(temp_label.shape)

        self.train_samples.append(temp_sample)
        self.train_labels.append(temp_label)

    #call function fit for each registered models
    def fit_models(self):
        for i in range(0, self.group_size):
            print("Training model %d" % (i+1))
            self.models[i].fit(self.train_samples[i], self.train_labels[i])


    #test function
    #compare prediction result of each sample with test label, then vote for final result of a test data
    def test(self, X, Y):
        test_size = len(Y)
        result_true = 0
        for i in range(0, test_size):
            class_dict = {}
            for model in self.models:
                y_hat = model.predict(X[i])
                if y_hat not in class_dict.keys() :
                    class_dict[y_hat] = 0

                class_dict[y_hat] += 1
            temp_result = max(class_dict, key=class_dict.get)
            if Y[i] == temp_result:
                result_true += 1
        accuracy = (result_true / test_size) * 100

        print("Test result : ")
        print("  - Group size : %d" % (self.group_size))
        print("  - Test size : %d" % (test_size))
        print("  - Accuracy : %.3f" % (accuracy))



