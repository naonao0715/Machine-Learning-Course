import numpy as np
from sklearn.preprocessing import OneHotEncoder

from scipy.special import softmax
import pandas as pd
import pdb
def Sigmoid(x):
    return  1.0 / (1.0 + np.exp(-x))


class LogisticRegresssion(object):


    def fit(self, X, Y, LR = 0.01, K = 10000):
        M = X.shape[0]
        bias_col = np.ones([M, 1])
        num_feature = X.shape[1]
        X = np.append(X, bias_col, 1) # M x F
        num_classes = len(np.unique(Y))
        self.W = np.zeros([num_classes, num_feature + 1]) # C x F
        self.enc = OneHotEncoder(handle_unknown='ignore')
        
        self.enc.fit(Y.reshape(-1, 1))
        Y_enc = self.enc.transform(Y.reshape(-1, 1)).toarray() # M x C

        cost_array = np.empty((0, num_classes))
        for i in range(K):
            logits_score = X @ self.W.T # M x C
            p = Sigmoid(logits_score)
            cost = - 1 / M * (Y_enc * np.log(p) + (1 - Y_enc) * np.log(1 - p))
            cost = np.sum(cost, axis = 0)
            cost_array = np.vstack([cost_array, cost])
            cost = np.sum(cost)

            delta =  LR / M * (p - Y_enc).T @ X
            self.W = self.W - delta
        return cost_array

    def predict(self, X):

        M = X.shape[0]
        bias_col = np.ones([M, 1])
        X = np.append(X, bias_col, 1) # M x F

        logits_score = X @ self.W.T # M x C
        prob = Sigmoid(logits_score)
        prob = prob > 0.5

        classes = np.argmax(prob, axis = 1)

        return classes


if __name__ == '__main__':
    df_1 = pd.read_csv('2F_2C.csv')

    X = df_1.iloc[:, :-1]
    Y = df_1.iloc[:, -1]

    lr = LogisticRegresssion2()
    
    cost = lr.fit(X.to_numpy(), Y.to_numpy(), K = 10000)
