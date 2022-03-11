"""
Author(s): Alejandro Hernandez
Last updated: March 11, 2022
Title: maxent.py


"""
import numpy as np

X: np.matrix  # matrix of ENTIRE survey space, 2D array of bioclimatic values
m: np.matrix  # matrix of sample space, 2D array of bioclimatic values
w: np.array  # array of lambdas (feature weights), 1D array
b: np.array  # array of betas (hyper-parameter), 1D array
f: np.array  # array of empirical feature averages
g: np.array  # array of expected feature averages
raw_prob: np.array  # array of raw probabilities over sample space
cum_prob: np.array  # array of cumulative probabilities over samples space


def empirical_feature_avg(survey_space: np.array) -> np.array:
    """The average value of each bioclimatic variables, calculated from our entire survey space"""
    # average each column of survey_space matrix
    return 0


def z_const(survey_space: np.array, feature_weights: np.array) -> float:
    """The Z constant assures that probabilities over a sample space sums to 1"""
    return np.sum([np.exp(np.dot(f_x, feature_weights)) for f_x in survey_space])


def gibbs(f_x: np.array, feature_weights: np.array, z_constant: float) -> float:
    """The Gibbs distribution that maximizes the likelihood of a sample space
    is our best estimate of the unknown probability distribution"""
    return np.exp(f_x, feature_weights) / z_constant


def log_loss(feature_weights: np.array, empirical_averages: np.array, z_constant: float, reg_params: np.array) -> float:
    """Regularized log log is the function we will minimize during the training process"""
    return np.dot(-feature_weights, empirical_averages) + np.log(z_constant) + np.dot(reg_params, np.abs(feature_weights))


def main():
    """Training process occurs here, done with sequential-update algorithm"""
    pass


if __name__ == "__main__":
    main()
