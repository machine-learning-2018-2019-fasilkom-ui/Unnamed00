import numpy as np
from LogReg import LogReg

class Bagging:
    def __init__(self):
        self.seed = None
        self.group_size = None
        self.train_samples = []
        self.train_labels = []
        self.models = []

    def generate_samples(self, X, Y, group_size = 10, seed_val = 585):
        self.seed = seed_val
        self.group_size = group_size

        np.random.seed(self.seed)
        for i in range(0, self.group_size):
            self.randomize_sample(X,Y)

        print("train samples : %d \ntrain labels : %d " % (len(self.train_samples), len(self.train_labels)))


    def randomize_sample(self, X, Y):
        indices = np.random.randint(0, len(Y), len(Y))
        temp_sample = X[indices]
        #print(temp_sample.shape)
        temp_label = Y[indices]
        #print(temp_label.shape)

        self.train_samples.append(temp_sample)
        self.train_labels.append(temp_label)

    def log_regression(sample, label):
        logreg = LogReg(False)
        logreg.fit(sample, label)
        return logreg

    def mlp(sample, label):
    #type = algo used for training
    #type = log_regression : deafult, logistic regression
    #type = svm : SVM
    #type = mlp : MLP, use library
    def fit_models(self, alg_type=log_regression):
        for i in range(0,self.group_size):
            print("Training model %d" % (i))
            self.models.append(alg_type(self.train_samples[i], self.train_labels[i]))


    #def svm(self, sample, label):
    #    svm = SVM()
    #    return svm
    #add test method for easier testing
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
        print("Group size : %d" % (self.group_size))
        print("Test size : %d" % (test_size))
        print("Accuracy : %.3f" % (accuracy))



