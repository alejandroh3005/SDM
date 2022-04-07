"""
Author(s): Alejandro Hernandez, Nilay Nagar
Last updated: March 28, 2022
Title: maxent.py
"""
import os
import numpy as np
import pandas as pd
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
    empirical_averages = get_empirical_averages(survey_space)
    n_features = len(empirical_averages)
    feature_weights = np.ones(n_features)
    reg_params = np.full((1,n_features), 0.1)

    Z = get_z_constant(survey_space=survey_space, feature_weights=feature_weights)

    """Training process occurs here, done with sequential-update algorithm"""
    MAX_ITER = 50
    MIN_CONVERGENCE = 0.0005
    STEP_SIZE = 0.1
    converged = False
    old_log_loss = get_reg_log_loss(feature_weights, empirical_averages, Z, reg_params)
    iter = 1
    while not converged and iter <= MAX_ITER:
        new_log_loss = get_reg_log_loss(feature_weights, empirical_averages, Z, reg_params)
        if iter > 1 and abs(old_log_loss - new_log_loss) < MIN_CONVERGENCE:
            converged = True
        elif iter >= MAX_ITER:
            break

        # Batch gradient descent
        gradient = [-weight*average + beta*abs(weight) for weight, average, beta in zip(feature_weights, empirical_averages, reg_params[0])]
        for j in range(n_features):
            delta = gradient[j] * STEP_SIZE
            feature_weights[j] -= delta
        print(f"\nIteration {iter}:")
        print(new_log_loss  )
        iter += 1



if __name__ == "__main__":
    main()
