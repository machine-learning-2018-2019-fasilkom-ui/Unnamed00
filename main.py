import numpy as np
import time
from Preprocessing import Preprocessing
from SVM import SVM
from LogReg import LogReg

pre = Preprocessing()
pre.read_data()
pre.reduce_data(0.99)
pre.split_data()
#svm = SVM()

#pre.training_set =  np.array([[1,7],[2,8],[3,8],[5,1],[6,-1],[7,3]])
#pre.training_label  = np.array([1,-1,-1,1,1,1])
start = time.time()
#svm.fit(pre.training_set, pre.training_label)
logreg = LogReg()
logreg.fit(pre.training_set, pre.training_label, 1000, 0.01, 'random' )

print(time.time() - start)
#print(svm.w, svm.b)
logreg.test(pre.test_set, pre.test_label)
