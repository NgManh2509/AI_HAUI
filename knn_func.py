import os
import numpy as np

class KNN:
    def __init__(self, K=3):
        self.K = K
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train, dtype="float32")
        self.y_train = np.array(y_train)

    def _predict_one(self, x):
        x = np.array(x, dtype="float32").reshape(1, -1)  
        
        dists = np.linalg.norm(self.X_train - x, axis=1) 

        idx = np.argsort(dists)[: self.K]
        k_labels = self.y_train[idx]

        values, counts = np.unique(k_labels, return_counts=True)
        return values[np.argmax(counts)]

    def predict(self, X_test):
        X_test = np.array(X_test, dtype="float32")
        if X_test.ndim == 1:
            return np.array([self._predict_one(X_test)])

        preds = [self._predict_one(x) for x in X_test]
        return np.array(preds)


def load_knn_from_npz(model_path):
    
    if not os.path.exists(model_path):
        return None

    data = np.load(model_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    k = int(data["k"][0]) if "k" in data.files else 5

    model = KNN(K=k)
    model.fit(X, y)
    return model
