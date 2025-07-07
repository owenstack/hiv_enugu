import numpy as np


def exponential_model(x, a, b, c):
    """Exponential growth model: y = a * exp(b * x) + c"""
    return a * np.exp(b * x) + c


def logistic_model(x, L, k, t0, c_offset):
    """
    Logistic growth model (S-curve).
    y = L / (1 + np.exp(-k * (x - t0))) + c_offset
    L: The maximum value (upper asymptote).
    k: The steepness of the curve (growth rate).
    t0: The x-value of the sigmoid's midpoint.
    c_offset: The offset (lower asymptote).
    """
    return L / (1 + np.exp(-k * (x - t0))) + c_offset


def richards_model(x, a, b, c, d, k_param): # Renamed k to k_param to avoid clash if we standardize it later
    """Richards growth model: y = a / (1 + np.exp(-b * (x - c)))**(1/d) + k_param"""
    # Note: Parameters here are not yet standardized like logistic/gompertz
    return a / (1 + np.exp(-b * (x - c))) ** (1 / d) + k_param


def gompertz_model(x, L, k, t0, c_offset):
    """
    Gompertz growth model.
    y = c_offset + (L - c_offset) * np.exp(-np.exp(-k * (x - t0)))
    L: The upper asymptote.
    k: The growth rate (steepness).
    t0: The x-value of the inflection point.
    c_offset: The lower asymptote.
    """
    # Amplitude of growth L_amp = L - c_offset
    # If L is defined as amplitude, then y = c_offset + L * np.exp(-np.exp(-k * (x - t0)))
    # The chosen form assumes L is the *absolute* upper asymptote.
    amplitude = L - c_offset
    return c_offset + amplitude * np.exp(-np.exp(-k * (x - t0)))
