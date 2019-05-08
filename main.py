import numpy as np
import time
from Preprocessing import Preprocessing
from SVM import SVM
from LogReg import LogReg
from Bagging import Bagging
from kernels import kernels
from LibSVM import LibSVM
from LibMLP import LibMLP
#run with bagging
#train_test_with_bagging(False)
#run without bagging
#train_test_without_bagging(False)


def preprocess_data(is_verbose, use_pca, threshold):
    start = time.time()
    pre = Preprocessing(verbose=is_verbose)
    pre.read_data()

    if use_pca :
        pre.reduce_data(threshold)
        print("  - Dimensionality reduction using PCA with threshold : %5f" %(threshold))

    pre.split_data() #split data with ratio 80:20

    print("Data preprocessing time : %f" %(time.time() - start))
    return pre

def train_test_without_bagging(verbose = False, use_pca=True, threshold = 0.999, mode = 0, num_iter = 1000):
    #num_iter = 1000 #number of iteration for training

    data = preprocess_data(verbose, use_pca, threshold)

    if mode == 0:
        model = create_models(1, num_iter, verbose, svm, kernels.polynomial, None, 3, 5)[0]
    elif mode == 1:
        model = create_models(1, num_iter, verbose, logistic_regression)[0]
    elif mode == 2:
        model = create_models(1, num_iter, verbose, libalg, libalg_type=0)[0]
    else:
        model = create_models(1, num_iter, verbose, libalg, libalg_type=1)[0]

    print("Training model without bagging")

    model.fit(data.training_set, data.training_label)

    test(model, data.test_set, data.test_label)

def test(model, test_set, test_label):
    true_count = 0
    for i in range(0, len(test_label)):
        y_hat = model.predict(test_set[i])
        if y_hat == test_label[i]:
            true_count += 1

    print("Test result :")
    print(" - Test size : %d" %(len(test_label)))
    print(" - Accuracy : %.3f" % (100 * true_count / len(test_label)))


def train_test_with_bagging(verbose = False, group_size = 10, use_pca = True, threshold = 0.999, mode=0, num_iter = 1000):
    N = group_size #number of models used
    #num_iter = 1000 #number of iteration for training each models

    data = preprocess_data(verbose, use_pca, threshold) #preprocess data (read, dimensional reduction, split)

    if mode == 0:
        models = create_models(N, num_iter, verbose, svm, kernels.polynomial, None, 3, 5)
    elif mode == 1:
        models = create_models(N, num_iter, verbose, logistic_regression)
    elif mode == 2:
        models = create_models(N, num_iter, verbose, libalg, libalg_type=0)
    else:
        models = create_models(N, num_iter, verbose, libalg, libalg_type=1)
    start = time.time()

    bagging = Bagging(models)
    bagging.generate_samples(data.training_set, data.training_label)

    print("Training %d models with %d iteration" %(N, num_iter))
    bagging.fit_models()

    print("Total training time : %f" %(time.time() - start))

    bagging.test(data.test_set, data.test_label)

#supported alg_type :
# - logistic_regression
# - SVM
# - MLP
#Note : to configure the model's parameters, modify the function of each alg_type provided
def create_models(N, num_iter, verbose, alg_type=None, kernel=kernels.linear, C=None, degree=1, sigma=1, libalg_type = 0):
    if alg_type is None:
        raise Exception("Please provide an algorithm type.\n")

    models = []
    for i in range(0, N):
        if alg_type == logistic_regression:
            models.append(logistic_regression(verbose, num_iter))
        elif alg_type == svm:
            models.append(svm(verbose, kernel, C, degree, sigma))
        elif alg_type == libalg:
            models.append(libalg(libalg_type))
        else:
            raise Exception("Unsupported algorithm type : %s" %(alg_type))

    return models

#use sklearn machine learning algorithm
# supported :
#  - SVM
#  - MLP Classifier
#solver_ = 'lbfgs', alpha_ = 1e-5, l_rate_ = 0.0001, layer_ = (5,2), random_state_ = 1, num_iter = 200):
def libalg(algorithm):
    if algorithm == 0:
        return LibSVM()
    elif algorithm == 1:
        return LibMLP()

#create a logistic regression model
def logistic_regression(is_verbose, num_iter):
    model = LogReg(verbose=is_verbose, num_iter=num_iter, mode = 'random', learning_rate = 0.0001)
    return model

#create an SVM model
#supported kernel :
# - linear kernel = linear_kernel
# - polynomial kernel = polynomial_kernel
# - rbf kernel = rbf
def svm(is_verbose, kernel, C, degree, sigma):
    model = SVM(kernel, C, degree, sigma)
    return model


train_test_with_bagging(False, 15, False, 0.999, 1, 10000)
#train_test_without_bagging(False, False, 0.999, 1, 10000)
train_test_with_bagging(False, 15, True, 0.999, 1, 10000)
#train_test_without_bagging(False, True, 0.999, 1, 10000)
