import numpy as np
from scipy.integrate import simpson
import tensorflow as tf


tf.config.experimental.enable_tensor_float_32_execution(False)


def generateWindows(a, bins):
    """
    Generate rolling windows from a full profile, which are suitable to be given to a neural functional.

    a: The one-body profile to be rearranged into windows (periodic boundary conditions are assumed), usually the density profile.
    bins: The number of bins that are taken into account at each side of the window. The total window will hence consist of (2 * bins + 1) histogram bins.
    """
    aPadded = np.pad(a, bins, mode="wrap")
    aWindows = np.empty((len(a), 2*bins+1))
    for i in range(len(a)):
        aWindows[i] = aPadded[i:i+2*bins+1]
    return aWindows


def c1(model, rho, params, c2=False):
    """
    Infer the one-body direct correlation profile from a given density profile with a given neural correlation functional.

    model: The neural correlation functional.
    rho: The density profile.
    params: A Dict with additional parameters that are required as input to the model (e.g. temperature, inverse box length).
    c2: If False, only return c1(x). If True, return both c1(x) as well as the corresponding two-body direct correlation function c2(x, x') which is obtained via autodifferentiation. If 'unstacked', give c2 as a function of x and x-x', i.e. as obtained naturally from the model.
    """
    inputBins = model.input["rho"].shape[1]
    windowBins = (inputBins - 1) // 2
    rhoWindows = generateWindows(rho, windowBins).reshape(rho.shape[0], inputBins)
    paramsInput = {key: tf.convert_to_tensor(np.full(rho.shape[0], value)) for key, value in params.items()}
    if c2:
        rhoWindows = tf.Variable(rhoWindows)
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(rhoWindows)
            result = model({"rho": rhoWindows, **paramsInput})["c1"]
        jacobiWindows = tape.batch_jacobian(result, rhoWindows).numpy().squeeze()
        c1_result = result.numpy().flatten()
        if c2 == "unstacked":
            c2_result = jacobiWindows
        else:
            c2_result = np.vstack([np.roll(np.pad(jacobiWindows[i], (0,rho.shape[0]-inputBins)), i-windowBins) for i in range(rho.shape[0])]).T
        return c1_result, c2_result
    return model.predict_on_batch({"rho": rhoWindows, **paramsInput})["c1"].flatten()


def c2x_bulk(model, rhobs, params, dx=0.01):
    """
    Evaluate the planar two-body direct correlation functional at multiple bulk densities.

    model: The neural correlation functional.
    rhobs: The values of the bulk densities.
    params: A Dict with additional parameters that are required as input to the model (e.g. temperature, inverse box length).
    dx: The spatial discretization interval.
    """
    inputBins = model.input["rho"].shape[1]
    xs = dx * np.linspace(-inputBins // 2, inputBins // 2, inputBins)
    rhobWindows = np.empty((len(rhobs), inputBins))
    for b, rhob in enumerate(rhobs):
        rhobWindows[b] = rhob
    rhoWindows = tf.Variable(rhobWindows)
    paramsInput = {key: tf.Variable(np.full((len(rhobs), 1), value)) for key, value in params.items()}
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(rhoWindows)
        result = model({"rho": rhoWindows, **paramsInput})
    c2x = tape.gradient(result, rhoWindows).numpy()
    return xs, c2x


def Fexc_funcintegral(model, rho, T, paramsWithoutT, n_alpha=100, dx=0.01):
    """
    Evaluate the excess free energy for a given density profile via functional line integration using a neural correlation functional.

    model: The neural correlation functional.
    rho: The target density profile.
    T: The temperature.
    paramsWithoutT: A Dict with additional parameters (without temperature) that are required as input to the model.
    n_alpha: The number of discretization points of the functional integration parameter alpha.
    dx: The spatial discretization interval.
    """
    alphas = np.linspace(0, 1, n_alpha)
    integrands = []
    for alpha in alphas:
        integrands.append(np.sum(rho * c1(model, alpha * rho, params={"T": T, **paramsWithoutT})))
    return -T * simpson(integrands, x=alphas) * dx


def calc_rho(model, T, paramsWithoutT, L, dx=0.01, Vext=lambda x: 0, mu=None, rho_mean=None, rho_init=lambda x: 0.5, alpha=0.02, tol=1e-5, maxiter=10000, print_every=0):
    """
    Attempt to determine the equilibrium density profile for the given parameters and external potential by solving the Euler-Langrange equation self-consistently with a mixed Picard iteration.

    model: The neural correlation functional.
    T: The temperature.
    paramsWithoutT: A Dict with additional parameters (without temperature) that are required as input to the model.
    L: The length of the system.
    dx: The spatial discretization interval of the profiles. Must match the model.
    Vext: The external potential, given as a callable function accepting the spatial location x. Constant zero if left unspecified.
    mu: The chemical potential (leave rho_mean unspecified).
    rho_mean: The mean density to be kept fixed (leave mu unspecified).
    rho_init: The initial density profile, given as a callable function accepting the spatial location x. If left unspecified, the system is initialized with constant bulk density 0.5. Provide here e.g. a step-like profile to investigate liquid-gas phase separation for low enough temperatures.
    alpha: The Picard mixing parameter.
    tol: The tolerance criterion for stopping the iteration.
    """
    if (mu is None and rho_mean is None) or (mu is not None and rho_mean is not None):
        raise ValueError("Specify either the chemical potential 'mu' or the mean density 'rho_mean'")
    xs = np.arange(dx/2, L, dx)
    Vext = Vext(xs)
    rho = np.zeros_like(xs)
    rho_new = np.zeros_like(xs)
    rho_reverse_new = np.zeros_like(xs)
    rho = rho_init(xs)
    valid = np.isfinite(Vext)
    rho[np.invert(valid)] = 0

    i = 1
    while True:
        rho_new[valid] = np.exp(-Vext / T + c1(model, rho, {"T": T, **paramsWithoutT}))[valid]
        rho_reverse_new[valid] = np.exp(-Vext / T + c1(model, rho[::-1], {"T": T, **paramsWithoutT}))[valid]
        rho_new *= 0.5
        rho_new += 0.5 * rho_reverse_new[::-1]
        if rho_mean is not None:
            rhob_new = np.sum(rho_new) * dx / L
            rho_new *= rho_mean / rhob_new
        if mu is not None:
            rho_new *= np.exp(mu / T)
        rho = (1 - alpha) * rho + alpha * rho_new
        delta = np.max(np.abs(rho_new - rho))
        i += 1
        if print_every > 0 and i % print_every == 0:
            print(f"Iteration {i}: delta = {delta}")
        if delta < tol:
            print(f"Converged after {i} iterations (delta = {delta})")
            return xs, rho
        if not np.isfinite(delta) or i >= maxiter:
            print(f"Did not converge after {i} iterations (delta = {delta})")
            return xs, None

