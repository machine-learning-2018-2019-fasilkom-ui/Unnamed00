from sklearn import svm

class LibSVM():
    def __init__(self, gamma_ = 'scale'):
        self.model = svm.SVC(gamma=gamma_)

    def fit(self, X, Y):
        self.model.fit(X, Y)
        print("Accuracy : %.3f" % (self.test(X, Y)))

    def predict(self, X):
        return self.model.predict(X.reshape(1,-1))[0]

    def test(self, X, Y):
        true_count = 0
        for i in range(0, len(Y)):
            y_hat = self.predict(X[i])
            if y_hat == Y[i]:
                true_count += 1

        return (100 * true_count / len(Y))
