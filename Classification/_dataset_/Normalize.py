from sklearn.preprocessing import MinMaxScaler
import numpy as np

def normalize(X):
    X=np.asarray(X)
    X=X.T
    scaler = MinMaxScaler().fit(X)
    X = scaler.transform(X)
    X = X.T
    return X