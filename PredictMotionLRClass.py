import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class MotionPredictor:
    def __init__(self, last_n=80, pred_int=1, min_movement=4.5, degree=1, future_n=50):
        self.last_n = last_n
        self.pred_int = pred_int
        self.min_movement = min_movement
        self.degree = degree
        self.future_n = future_n

    def predict(self, data):
        if len(data) > self.last_n:
            data = data[-1 * self.last_n::self.pred_int]
            data = np.array(data)
            X = data[:, 0].reshape(-1, 1)
            Y = data[:, 1]
            poly = PolynomialFeatures(degree=self.degree, include_bias=False)
            X = poly.fit_transform(X)
            model = LinearRegression().fit(X, Y)
            print(model.coef_, model.intercept_)
            if np.sum(np.diff(data[:, 0])) > self.min_movement:
                X = np.array([[i] for i in range(int(data[-1, 0]), int(data[-1, 0] + self.future_n), 1)], dtype='float32')
                X = poly.transform(X)
                preds = model.predict(X).reshape(-1, 1)
                X = X[:, 0].reshape(-1, 1)
                preds = np.concatenate([X, preds], axis=-1)
                return preds
            elif np.sum(np.diff(data[:, 0])) < -1 * self.min_movement:
                X = np.array([[i] for i in range(int(data[-1, 0]), int(data[-1, 0] - self.future_n), -1)], dtype='float32')
                X = poly.transform(X)
                preds = model.predict(X).reshape(-1, 1)
                X = X[:, 0].reshape(-1, 1)
                preds = np.concatenate([X, preds], axis=-1)
                return preds
        preds = np.array([[None, None] for _ in range(self.future_n)], dtype='float32')
        return preds

if __name__ == '__main__':
    x = np.random.random_integers(-100, 100, (100, 1))
    y = 5 * x + np.random.random_integers(0, 10, (100, 1))
    data = np.concatenate([x, y], axis=-1)
    model = MotionPredictor(last_n=10)
    preds = model.predict(data)
    print(data)
    print(preds)
