import numpy as np
from sklearn.linear_model import LinearRegression

def add_noise(y_pred, r_squared):
    """
    Add random noise to y_true to achieve the desired R squared value.
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


def extract_coef(X, y, X_feature_names, decimal_places):

    # Create a LinearRegression object
    lr = LinearRegression()

    # Fit the model using the training data
    lr.fit(X, y)

    # Extract the coefficient values
    coef_list = list(np.round(lr.coef_, decimal_places))

    # Print the coefficient values
    print("Coefficient values:")
    for feature_name, coef in zip(X_feature_names, coef_list):
        print(feature_name + "= " + str(coef))
    return
