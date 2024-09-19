import numpy as np
from sklearn.linear_model import LinearRegression

def add_noise(y_pred, r_squared):
    """
    Adds random noise to y to achieve the desired R squared value.
    :param y_pred: y variable noise should be added to
    :param r_squared: desired R-squared value y should be predicted from X
    :return: y: y_pred with noise
    :return: iter: number of iterations
    """
    n = y_pred.shape[0] #The number of observations or data points in y_pred
    v = np.sum((y_pred - np.mean(y_pred)) **2) #variance: sum of the squared differences between each value in y_pred and the mean of y_pred
    u = v * (1 - r_squared) / r_squared #scaled variance by the desired r_squared
    noise_var = u / n #The variance of the noise to be added to y_pred
    noise = np.random.normal(0, np.sqrt(noise_var), size=n) # normal distribution on 0, with SD (sqrt of noise_var)
    iter = 0
    while np.mean(noise) > 0.01 or -0.01 > np.mean(noise) or np.std(noise) > np.sqrt(noise_var) + 0.01 or np.sqrt(noise_var) - 0.01 > np.std(noise):
        noise = np.random.normal(0, np.sqrt(noise_var), size=n)
        iter = iter + 1
    y = y_pred + noise
    print("noise added: " + str(round(noise_var, 2)))
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


def generate_interaction_data(n_points=1000, noise_std=0.5, z_influence=5):
    """
    Generates synthetic data where the relationship between X and y is influenced by an interaction with Z.
    The direction of the parabolic relationship (whether it opens upward or downward) depends on the value of Z.
    :param n_points: int, optional (default=1000)
        Number of data points to generate.
    :param noise_std: float, optional (default=0.5)
        Standard deviation of the noise added to the y values.
    :param z_influence: float, optional (default=5)
        Standard deviation of Z, which influences whether the parabola opens up or down.
    :return: X: numpy.ndarray
        Array of X values (independent variable) ranging from 1 to 100.
    :return: y: numpy.ndarray
        Array of y values (dependent variable) generated as a function of X and Z, with added noise.
    :return: Z: numpy.ndarray
        Array of Z values that determine whether the parabola opens up or down.
    """
    # Create X as a sequence
    X = np.linspace(1, 100, n_points)

    # Generate Z, which will influence whether the parabola opens up or down
    Z = np.random.normal(0, z_influence, size=n_points)

    # The interaction: the sign of Z determines the direction of the parabola
    y = np.where(Z >= 0,
                 0.01 * (X - 50) ** 2 + Z,  # Parabola opens upwards if Z >= 0
                 -0.01 * (X - 50) ** 2 + Z)  # Parabola opens downwards if Z < 0

    # Add some noise to the data
    y += np.random.normal(0, noise_std, size=n_points)

    return X, y, Z