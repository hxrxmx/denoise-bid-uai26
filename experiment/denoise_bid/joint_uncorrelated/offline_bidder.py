import numpy as np

from experiment.denoise_bid.ctr_only.gmm import fit_gmm
from experiment.denoise_bid.ctr_only.expectation import ctr_expectation
from experiment.denoise_bid.ctr_only.lp import solve_dual
from experiment.denoise_bid.ctr_only.bid import bids


def uncorrelated_denoise_bid(
    config,
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
        size=100 * n_components,
        replace=False
    )
    ctr_gmm_logit = ctr_logit[indexes]
    cvr_gmm_logit = cvr_logit[indexes]
    ctr_gmm_sigma = sigma_ctr[indexes]
    cvr_gmm_sigma = sigma_cvr[indexes]

    weights, means, sigmas = fit_gmm(
        config,
        ctr_gmm_logit,
        ctr_gmm_sigma,
        n_components,
    )
    ctr = ctr_expectation(
        ctr_logit,
        sigma_ctr,
        weights,
        means,
        sigmas,
    )

    weights, means, sigmas = fit_gmm(
        config,
        cvr_gmm_logit,
        cvr_gmm_sigma,
        n_components,
    )
    cvr = ctr_expectation(
        cvr_logit,
        sigma_cvr,
        weights,
        means,
        sigmas,
    )

    p, q = solve_dual(config, ctr, cvr, wp, budget, target_cpc)
    bids_res = bids(ctr, cvr * ctr, p, q, target_cpc)
    return bids_res
