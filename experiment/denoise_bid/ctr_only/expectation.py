import numpy as np
from experiment.utils.utils import sigmoid
from scipy.special import expit


def ctr_expectation(
    ctr_logit,
    ctr_logit_sigma,
    ctr_weights,
    ctr_means,
    ctr_sigmas,
):
    L = ctr_logit[:, np.newaxis]
    S2 = ctr_logit_sigma[:, np.newaxis]**2
    
    W = ctr_weights[np.newaxis, :]
    M = ctr_means[np.newaxis, :]
    SI2 = ctr_sigmas[np.newaxis, :]**2

    sum_S2 = S2 + SI2
    
    inv_S2 = 1.0 / S2
    inv_SI2 = 1.0 / SI2
    sum_inv = inv_S2 + inv_SI2

    mu_eff = (L * inv_S2 + M * inv_SI2) / sum_inv
    sigma_sq_eff = 1.0 / sum_inv

    multiplier = np.exp(-0.5 * (L - M)**2 / sum_S2)
    denom_term = (W * multiplier) / np.sqrt(sum_S2)

    sigmoid_int_approx = expit(
        mu_eff / np.sqrt(1 + (np.pi / 8) * sigma_sq_eff)
    )

    numerator = np.sum(denom_term * sigmoid_int_approx, axis=1)
    denominator = np.sum(denom_term, axis=1)
    
    return numerator / (denominator + 1e-10)
