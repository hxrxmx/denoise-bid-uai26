import numpy as np
from experiment.utils.utils import sigmoid


def ctr_expectation(
    ctr_logit,
    ctr_logit_sigma,
    ctr_weights,
    ctr_means,
    ctr_sigmas,
):
    ctr_exp = np.zeros_like(ctr_logit)

    for i, (logit, sigma) in enumerate(
        zip(ctr_logit, ctr_logit_sigma)
    ):
        numerator = 0.0
        denominator = 0.0

        for w_i, mu_i, sigma_i in zip(ctr_weights, ctr_means, ctr_sigmas):
            mu_eff = (logit / sigma ** 2 + mu_i / sigma_i ** 2) / \
                (1 / sigma**2 + 1 / sigma_i ** 2)
            sigma_sq_eff = 1 / (1 / sigma**2 + 1 / sigma_i**2)
            multiplier = np.exp(
                -(logit - mu_i)**2 / (2 * (sigma**2 + sigma_i**2))
            )

            denom_term = (
                w_i * multiplier
                / np.sqrt(sigma**2 + sigma_i**2)
            )

            sigmoid_int_approximation = sigmoid(
                mu_eff / (np.sqrt(1 + np.pi / 8 * sigma_sq_eff))
            )

            denominator += denom_term
            numerator += denom_term * sigmoid_int_approximation

        ctr_exp[i] = numerator / (denominator + 1e-10)
    return ctr_exp
