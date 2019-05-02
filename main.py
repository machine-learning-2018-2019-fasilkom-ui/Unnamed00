import numpy as np
import time
from Preprocessing import Preprocessing
from SVM import SVM
from LogReg import LogReg
from Bagging import Bagging
pre = Preprocessing()
pre.read_data()
pre.reduce_data(0.995)
pre.split_data()
#svm = SVM()

#pre.training_set =  np.array([[1,7],[2,8],[3,8],[5,1],[6,-1],[7,3]])
#pre.training_label  = np.array([1,-1,-1,1,1,1])
start = time.time()
#svm.fit(pre.training_set, pre.training_label)
#logreg = LogReg()
#logreg.fit(pre.training_set, pre.training_label, 1000, 0.01, 'default' )
bagging = Bagging()
bagging.generate_samples(pre.training_set, pre.training_label)
bagging.fit_models()
print(time.time() - start)
#print(svm.w, svm.b)
#logreg.test(pre.test_set, pre.test_label)
bagging.test(pre.test_set, pre.test_label)
#print("Accuracy : %.3f" % (acc))
