import numpy as np
import time
from Preprocessing import Preprocessing
from SVM import SVM
from LogReg import LogReg
from Bagging import Bagging

#run with bagging
#train_test_with_bagging(False)
#run without bagging
#train_test_without_bagging(False)


def preprocess_data(is_verbose, use_pca=True, threshold=0.999):
    start = time.time()
    pre = Preprocessing(verbose=is_verbose)
    pre.read_data()

    if use_pca :
        pre.reduce_data(threshold)
        print("  - Dimensionality reduction using PCA with threshold : %5f" %(threshold))

    pre.split_data() #split data with ratio 80:20

    print("Data preprocessing time : %f" %(time.time() - start))
    return pre

def train_test_without_bagging(verbose = False):
    num_iter = 1000 #number of iteration for training
    use_pca = True
    pca_threshold = 0.995

    preprocessed_data = preprocess_data(verbose, use_pca, pca_treshold)


def train_test_with_bagging(verbose = False):
    N = 10 #number of models used
    num_iter = 2000 #number of iteration for training each models
    use_pca = True
    pca_treshold = 0.995

    data = preprocess_data(verbose, use_pca, pca_treshold) #preprocess data (read, dimensional reduction, split)

    models = create_models(N, num_iter, verbose, logistic_regression)
    start = time.time()

    bagging = Bagging(models)
    bagging.generate_samples(data.training_set, data.training_label)

    print("Training %d models with %d epoch" %(N, num_iter))
    bagging.fit_models()

    print("Total training time : %f" %(time.time() - start))

    bagging.test(data.test_set, data.test_label)

#supported alg_type :
# - logistic_regression
# - SVM
# - MLP
#Note : to configure the model's parameters, modify the function of each alg_type provided
def create_models(N, num_iter, verbose, alg_type=None):
    if alg_type is None:
        raise Exception("Please provide an algorithm type.\n")

    models = []
    for i in range(0, N):
        models.append(alg_type(verbose, num_iter))

    return models

#create a logistic regression model
def logistic_regression(is_verbose, num_iter):
    model = LogReg(verbose=is_verbose, num_epoch=num_iter)
    return model

#create an SVM model
def svm(is_verbose, num_iter):
    model = SVM()
    return model

#create an MLP model
def mlp(is_verbose, num_iter):
    model = 1
    return model

train_test_with_bagging(False)
