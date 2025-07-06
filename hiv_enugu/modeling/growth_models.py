import numpy as np


def exponential_model(x, a, b, c):
    """Exponential growth model: y = a * exp(b * x) + c"""
    return a * np.exp(b * x) + c


def logistic_model(x, a, b, c, d):
    """Logistic growth model: y = a / (1 + np.exp(-b * (x - c))) + d"""
    return a / (1 + np.exp(-b * (x - c))) + d


def richards_model(x, a, b, c, d, k):
    """Richards growth model: y = a / (1 + np.exp(-b * (x - c)))**(1/d) + k"""
    return a / (1 + np.exp(-b * (x - c))) ** (1 / d) + k


def gompertz_model(x, a, b, c, d):
    """Gompertz growth model: y = a * np.exp(-b * np.exp(-c * x)) + d"""
    return a * np.exp(-b * np.exp(-c * x)) + d
