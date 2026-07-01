import numpy as np
from experiment.risk_bid.lp import solve_dual


def risk_bid(
    config,
    ctr_noised_logit,
    cvr_noised_logit,
    wp,
    budget,
    target_cpc,
    sigma_ctr,
    alpha,
):
    ctr_noised = 1 / (1 + np.exp(-ctr_noised_logit))
    cvr_noised = 1 / (1 + np.exp(-cvr_noised_logit))

    return bids(
        config,
        ctr_noised,
        cvr_noised,
        wp,
        budget,
        sigma_ctr,
        alpha,
    )


def bids(config, ctr, cvr, wp, budget, sigma_ctr, alpha=0.1):
    std_ctr = sigma_ctr * (ctr * (1 - ctr))

    p_val = solve_dual(config, ctr, cvr, wp, budget, None, sigma_ctr, alpha)

    phi = 1.0 / (p_val + 1e-12)

    bids = phi * cvr * (ctr - alpha * std_ctr)

    return np.maximum(bids, 0.0)