import numpy as np

from experiment.denoise_bid.joint.expectation \
    import ctr_expectation, value_expectation
from experiment.denoise_bid.joint.lp import solve_dual
from experiment.denoise_bid.joint.gmm import fit_gmm
from experiment.denoise_bid.joint.bid import bids


def denoise_bid(
    ctr_logit,
    cvr_logit,
    sigma_ctr,
    sigma_cvr,
    wp,
    budget,
    target_cpc,
    n_components,
):
    indexes = np.random.choice(
        len(ctr_logit),
        size=200 * n_components,
        replace=False
    )
    ctr_gmm_logit = ctr_logit[indexes]
    cvr_gmm_logit = cvr_logit[indexes]
    ctr_gmm_sigma = sigma_ctr[indexes]
    cvr_gmm_sigma = sigma_cvr[indexes]
    weights, means, sigmas = fit_gmm(
        ctr_gmm_logit,
        cvr_gmm_logit,
        ctr_gmm_sigma,
        cvr_gmm_sigma,
        n_components,
    )

    ctr = ctr_expectation(
        ctr_logit,
        cvr_logit,
        sigma_ctr,
        sigma_cvr,
        weights,
        means,
        sigmas,
    )
    value = value_expectation(
        ctr_logit,
        cvr_logit,
        sigma_ctr,
        sigma_cvr,
        weights,
        means,
        sigmas,
    )

    p, q = solve_dual(ctr, value, wp, budget, target_cpc)
    return bids(ctr, value, p, q, target_cpc)
