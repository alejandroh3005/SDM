"""
Author(s): Alejandro Hernandez
Last updated: May 3, 2022
Title: maxent.py
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class MaxEnt:
    def __init__(self, **kwargs):
        self.features: np.array(np.array) = None  # features values at sample
        self.empirical_avg: np.array(np.array) = None  # empirical/arithmetic averages of features f1,...,fn

        # normalize features f1,...,fn so that fj:X -> [0,1]
        scaler = MinMaxScaler()
        scaler.fit(X=self.features)
        self.norm_features: np.array = scaler.transform(X=self.features)

        # set hyper-parameters of maxent training
        self.reg_params: np.array = np.std(self.features)
        self.weights = [0]*len(self.features)

    def predict(self, x) -> np.array:
        """
        Gibbs distribution takes feature weights and feature values to calculate a probability
        :return:
        """
        w = self.weights
        raw_probs = [np.exp(np.dot(w, x_i)) for x_i in x]
        return raw_probs / np.sum(raw_probs)

    def expected_value(self, x):
        p = self.predict(x)
        return np.dot(x,p)

    def func_j(self, delta, j, exp_fj) -> float:
        w = self.weights
        b = self.reg_params
        emp = self.empirical_avg
        a = -delta * emp[j]
        b = np.log((1 + exp_fj * (np.exp(delta) - 1)))
        c = b[j] * (np.abs(w[j] + delta) + np.abs(w[j]))
        return a + b + c

    def fit(self, x, MAX_ITER=500):
        self.sample = x
        self.n_iter = 0
        for t in range(MAX_ITER):
            # find (j,d) to minimize Fj
            min_j, min_Fj, min_d = 0, 0, 0

            # for all possible feature weights
            for j in range(len(self.weights)):
                exp_fj = expected_value(self.features[j])
                # calculate all candidate options of delta
                d1 = ...
                d2 = ...
                d3 = -self.weights[j]
                for d in [d1,d2,d3]:
                    if self.weights[j] += d: continue	# weights cannot be negative, try next delta value
                    fj = Fj(d,j,exp_fj)
                    if fj < minFj:
                        min_Fj = fj
                        min_j = j
                        min_d = d	# only recorded for debugging/curiousity
            # update selected weight
            self.weights[j] += min_d
            self.n_iter += 1


parent_dir = r"D:"
data_dir = f"{parent_dir}\output\madagascar\maxent_input"

def get_empirical_averages(survey_space: pd.DataFrame) -> pd.Series:
    """The average value of each bioclimatic variable, calculated from our entire survey space"""
    return survey_space.mean().loc["bio_1":]

def get_z_constant(survey_space: pd.DataFrame, feature_weights: list) -> float:
    """The Z constant assures that probabilities over a sample space sums to 1"""
    return np.sum([np.exp(np.dot(f_x.loc["bio_1":], feature_weights)) for _, f_x in survey_space.iterrows()])

def get_gibbs_prob(f_x: pd.Series, feature_weights: list, z_constant: float) -> float:
    """The Gibbs distribution that maximizes the likelihood of a sample space
    is our best estimate of the unknown probability distribution"""
    return np.exp(np.dot(f_x, feature_weights)) / z_constant

def get_reg_log_loss(feature_weights: np.array, empirical_averages: pd.Series, z_constant: float, reg_params: list) -> float:
    """Regularized log loss is the function we will minimize during the training process"""
    return np.dot(-feature_weights, empirical_averages) + np.log(z_constant) + np.dot(reg_params, np.abs(feature_weights))

def change_in_loss(empirical_averages:[], expected_averages:[], w:[], delta:float, j:int, beta:[]):
    """TODO: complete"""
    return delta * empirical_averages[j] + np.log(1 + (np.exp(delta) - 1) * expected_averages[j]) + w[j] * (abs(w+delta)-abs(w[j]))

def main():
    X: np.matrix  # matrix of ENTIRE survey space, 2D array of bioclimatic values
    m: np.matrix  # matrix of sample space, 2D array of bioclimatic values
    w: np.array  # array of lambdas (feature weights), 1D array
    b: np.array  # array of betas (hyper-parameter), 1D array
    f: np.array  # array of empirical feature averages
    g: np.array  # array of expected feature averages
    raw_prob: np.array  # array of raw probabilities over sample space
    cum_prob: np.array  # array of cumulative probabilities over samples space

    survey_space = pd.read_csv(data_dir + "\climate_data.csv")
    emp_avg = get_empirical_averages(survey_space)
    n_features = len(emp_avg)
    feature_weights = np.ones(n_features)
    reg_params = np.full((1,n_features), 0.1)

    Z = get_z_constant(survey_space=survey_space, feature_weights=feature_weights)

    """Training process occurs here, done with sequential-update algorithm"""
    MAX_ITER = 50
    MIN_CONVERGENCE = 0.0005
    # for t=1,2,3,...
    for j in range(MAX_ITER):
        # find delta and j that maximize change in loss
        option_a = np.log(emp_avg[j])
