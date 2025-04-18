import numpy as np
import matplotlib.pyplot as plt




def entropy(psi: callable):
    probability_density = np.abs(psi)**2
    return -np.sum(probability_density * np.log(probability_density + 1e-10))

def psi(x, t):
    # Example wave function: a Gaussian wave packet
    return np.exp(-0.5 * (x - t)**2) * np.exp(1j * t * x)
