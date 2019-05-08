from sklearn.neural_network import MLPClassifier

class LibMLP():
    def __init__(self, solver_ = 'lbfgs', alpha_ = 1e-5, l_rate_ = 0.0001, layer_ = (5,2), random_state_ = 1, num_iter = 1000):
        self.model = MLPClassifier(solver=solver_, alpha = alpha_, learning_rate_init = l_rate_, hidden_layer_sizes=layer_, random_state=random_state_, max_iter = num_iter)

    def fit(self, X, Y):
        self.model.fit(X, Y)
        print("Accuracy : %.3f" %(self.test(X, Y)))

    def predict(self, X):
        return self.model.predict(X.reshape(1,-1))[0]

    def test(self, X, Y):
        true_count = 0
        for i in range(0, len(Y)):
            y_hat = self.predict(X[i])
            if y_hat == Y[i]:
                true_count += 1

        return (100 * tru_count / len(Y))
