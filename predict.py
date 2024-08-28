import keras
import numpy as np

from utils import calc_rho


model = keras.models.load_model(f"models/LJ_T1.0-2.0_L5-20.keras")

L = 100
dx = 0.01
T = 0.9
rho_mean = 0.35

def rho_init(x):
    return 0.5*np.tanh(L/4-x) + 0.5 + 0.5*np.tanh(x-3*L/4) + 0.5

xs, rho = calc_rho(model, T, {"L_inv": 0}, L, dx, rho_mean=rho_mean, rho_init=rho_init, print_every=100)
print(rho)
