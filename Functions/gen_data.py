import numpy as np
from sklearn.linear_model import LinearRegression

def add_noise(y_pred, r_squared):
    """
    Adds random noise to y to achieve the desired R squared value.
    :param y_pred: y variable noise should be added to
    :param r_squared: desired R-squared value y should be predicted from X
    :return: y_red with noise, number of iterations
    """
    n = y_pred.shape[0]
    v = np.sum((y_pred - np.mean(y_pred)) **2)
    u = v * (1 - r_squared) / r_squared
    noise_var = u / n
    noise = np.random.normal(0, np.sqrt(noise_var), size=n)
    iter = 0
    while np.mean(noise) > 0.01 or -0.01 > np.mean(noise) or np.std(noise) > np.sqrt(noise_var) + 0.01 or np.sqrt(noise_var) - 0.01 > np.std(noise):
        noise = np.random.normal(0, np.sqrt(noise_var), size=n)
        iter = iter + 1
    y = y_pred + noise
    return y, iter


def extract_coef(X, y, X_feature_names, decimal_places,
                 file_path="Outputs/",
                 file_name="coefficients.txt"):
    """
    Extracts and saves the regression coefficients from a fitted model
    :param X: X dataframe
    :param y: y dataframe
    :param X_feature_names: list of strings containing feature names
    :param decimal_places: integer, number of decimal places for rounding
    :param file_path: string for file path to save to
    :param file_name: string for file save name
    """
    # Create a LinearRegression object
    lr = LinearRegression()

    # Fit the model using the training data
    lr.fit(X, y)

    # Extract the coefficient values
    coef_list = list(np.round(lr.coef_, decimal_places))

    # Print the coefficient values
    with open(file_path+file_name, "w") as txt:
        txt.write("Coefficient values:\n")
        for feature_name, coef in zip(X_feature_names, coef_list):
            txt.write(f"{feature_name} = {coef}\n")
    return
