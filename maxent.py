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
    return delta * emp_avg + np.log(1 + (np.exp(delta) - 1) * exp_val) + beta * (abs(weight + delta) - abs(weight))

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

    def fit(self, MAX_ITER=500):
        emp = self.empirical_avg
        w = self.weights
        b = self.reg_params
        for iter in range(MAX_ITER):
            # iteratively find the feature j (j) and change in feature j (d)
            # that minimizes change in regularized log loss
            min_j, min_d, min_Fj = 0, 0, 0
            # for all possible features, test candidate values of delta
            # and keep track of the that pair maximizes Fj
            for j in range(len(w)):
                exp_fj = self.expected_value(w=w, f_j=self.features[:,j])
                deltas = [0,0, -w[j]]     # calculate the 3 candidate values of delta
                deltas[0] = 0   # TODO compute candidate delta values
                deltas[1] = 0
                # remove all invalid candidates
                if w[j] + deltas[0] < 0: deltas[0] = 0
                if w[j] + deltas[1] > 0: deltas[1] = 0
                # from all pairs of delta and j, compute minimum change in reg. log loss
                for i, d in enumerate(deltas):
                    if d == 0: continue     # ignore delta values of zero
                    if w[j] + d < 0: exit("Features weights must be non-negative.")
                    fj = change_in_loss(emp_avg=emp[j], exp_val=exp_fj, weight=w[j], delta=d, beta=b[j])
                    if fj < min_Fj:
                        min_Fj = fj
                        min_j = j
                        min_d = i   # only recorded for debugging/curiosity
                # end for
            # update weight to minimize change in regularized log loss
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

    # predictions = maxent.predict(w=maxent.weights)
    # feature_expectations = maxent.expected_value(w=maxent.weights, f_j=maxent.features[:,1])

    # fit the model
    maxent.fit()
    print(maxent.weights)
    print(maxent.empirical_avg)
    print(maxent.reg_params)
    print(maxent.features[0])


if __name__ == '__main__':
    main()