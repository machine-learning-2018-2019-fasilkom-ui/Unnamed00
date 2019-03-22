import numpy as np
from Preprocessing import Preprocessing
from SVM import SVM

pre = Preprocessing()
pre.read_data()
pre.reduce_data(0.97)
pre.split_data()
print(pre.full_data.shape)
svm = SVM()

training_set =  np.array([[1,7],[2,8],[3,8],[5,1],[6,-1],[7,3]])
training_label  = np.array([-1,-1,-1,1,1,1])

svm.fit(pre.training_set, pre.training_label)

print(svm.w, svm.b)
print(svm.predict(pre.test_data[0:], pre.test_label[0]))
