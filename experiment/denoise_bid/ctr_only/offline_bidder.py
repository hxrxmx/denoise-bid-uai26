import numpy as np

from experiment.denoise_bid.ctr_only.gmm import fit_gmm
from experiment.denoise_bid.ctr_only.expectation import ctr_expectation
from experiment.denoise_bid.ctr_only.lp import solve_dual
from experiment.denoise_bid.ctr_only.bid import bids
from experiment.utils.utils import sigmoid


def denoise_bid(
    config,
    ctr_logit,
    ctr_sigma,
    cvr_logit,
    cvr_sigma,
    wp,
    budget,
    target_cpc,
    n_components,
):
    indexes = np.random.choice(
        len(ctr_logit),
        size=100 * n_components,
        replace=False,
    )
    ctr_gmm_logit = ctr_logit[indexes]
    ctr_gmm_sigma = ctr_sigma[indexes]
    weights, means, sigmas = fit_gmm(
        config,
        ctr_gmm_logit,
        ctr_gmm_sigma,
        n_components,
    )
    ctr = ctr_expectation(ctr_logit, ctr_sigma, weights, means, sigmas)
    cvr = sigmoid(cvr_logit)
    p, q = solve_dual(config, ctr, cvr, wp, budget, target_cpc)

    return bids(ctr, cvr, p, q, target_cpc)
