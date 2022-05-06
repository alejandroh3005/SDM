"""
Author(s): Alejandro Hernandez
Last updated: May 5, 2022
Title: maxent.py
"""
import numpy as np
import pandas as pd

def regularize_features(values: np.array, min=0, max=1) -> np.array:
    feature_std = (values - values.min(axis=0)) / (values.max(axis=0) - values.min(axis=0))
    return feature_std * (max - min) + min

def change_in_loss(emp_avg: float, exp_val: float, weight: float, delta: float, beta: float) -> float:
    a = delta * emp_avg
    b = np.log(1 + (np.exp(delta) - 1) * exp_val)
    c = beta * (abs(weight + delta) - abs(weight))
    chg_lss = a + b + c
    # delta is nan
    # print(delta)
    return chg_lss

# def get_reg_log_loss(feature_weights:np.array, empirical_averages:pd.Series, z_constant:float, reg_params:list) -> float:
#    """Regularized log loss is the function we will minimize during the training process"""
#    return np.dot(-feature_weights, empirical_averages) + np.log(z_constant) + np.dot(reg_params, np.abs(feature_weights))

# def get_z_constant(survey_space:pd.DataFrame, feature_weights:list) -> float:
#    """The Z constant assures that probabilities over a sample space sums to 1"""
#    return np.sum([np.exp(np.dot(f_x.loc["bio_1":], feature_weights)) for _, f_x in survey_space.iterrows()])


class MaxEnt:
    def __init__(self, **kwargs):
        presence = kwargs['presence_df']
        raw_features: np.array(np.array) = np.array(presence.drop(['pointid'],axis=1))  # features values of sample points
        self.features = regularize_features(values=raw_features)    # normalize features f1,...,fn so that fj:X -> [0,1]
        self.empirical_avg: np.array = self.features.mean(axis=0)  # empirical/arithmetic averages of features f1,...,fn
        # set hyper-parameters of maxent training
        self.reg_params: np.array = np.std(self.features, axis=0)
        self.weights = [0] * self.features[0]

    def predict(self, w:np.array) -> np.array:
        """ Gibbs distribution takes feature weights and feature values to calculate a probability
        :return: array of probabilities"""
        probabilities = [np.exp(np.dot(w, x_i)) for x_i in self.features]
        return probabilities / np.sum(probabilities)

    def expected_value(self, w: np.array, f_j: np.array) -> float:
        """ :return: the expected value of feature j """
        p = self.predict(w)
        return np.dot(p, f_j)

    def fit(self, MAX_ITER=50):
        emp = self.empirical_avg
        w = self.weights
        b = self.reg_params
        for iter in range(MAX_ITER):
            # iteratively find the feature j (j) and change in feature j (d)
            # that minimizes change in regularized log loss
            min_j, min_d, min_Fj = 0, 0, None
            # for all possible features, test candidate values of delta
            # and keep track of the that pair maximizes Fj
            for j in range(len(w)):
                exp_fj = self.expected_value(w=w, f_j=self.features[:,j])
                deltas = [None, None, -w[j]]  # calculate the 3 candidate values of delta
                deltas[0] = 0  # np.log((emp[j] - b[j]) * (1 - exp_fj) / (exp_fj * (1 - emp[j] + b[j])))
                deltas[1] = 0  # np.log((emp[j] + b[j]) * (1 - exp_fj) / (exp_fj * (1 - emp[j] - b[j])))
                # remove all invalid candidates
                if w[j] + deltas[0] < 0 or np.isnan(deltas[0]): deltas[0] = 0
                if w[j] + deltas[1] > 0: deltas[1] = 0
                # from all pairs of delta and j, compute minimum change in reg. log loss
                for i, d in enumerate(deltas):
                    # if d == 0: continue     # ignore delta values of zero
                    fj = change_in_loss(emp_avg=emp[j], exp_val=exp_fj, weight=w[j], delta=d, beta=b[j])
                    if j == 0:
                        min_Fj = fj
                    if fj <= min_Fj:
                        min_Fj = fj
                        min_j = j
                        min_d = d
                # end for
            # update weight to minimize change in regularized log loss
            # print(f"Iteration :{iter}: changed feature {min_j} from {w[min_j]} to {w[min_j] + min_d}")
            w[min_j] += min_d
        # end for
        # Once maximum iterations have been reached, set class weights to learned weights
        self.weights = w

def main():
    data_directory = "D:\\output\\madagascar\\maxent_input"
    presence_df = pd.read_csv(f"{data_directory}\\point_data.csv")
    # drop unnecessary columns and convert to 2D numpy array
    presence_df = presence_df[['pointid'] + [f'bio_{i}' for i in range(1, 11)]]
    maxent = MaxEnt(presence_df=presence_df)

    # feature_expectations = maxent.expected_value(w=maxent.weights, f_j=maxent.features[:,1])

    # fit the model
    maxent.fit()
    predictions = maxent.predict(w=maxent.weights)

    print(predictions)
    print(maxent.weights)


if __name__ == '__main__':
    main()