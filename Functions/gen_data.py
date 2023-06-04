import numpy as np

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