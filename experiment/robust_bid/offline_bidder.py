import numpy as np

from experiment.robust_bid.lp import solve_dual
from experiment.robust_bid.bid import bids
from experiment.utils.utils import sigmoid


def robust_bid(
    ctr_logit,
    cvr_logit,
    sigma_ctr,
    sigma_cvr,
    wp,
    budget,
    target_cpc,
    epsilon=None,
    robust_k=3,
):
    if epsilon is None:
        epsilon = np.sum(
            (
                sigmoid(ctr_logit + sigma_ctr * robust_k) -
                sigmoid(ctr_logit - sigma_ctr * robust_k)
            )**2
        ) / 2
    ctr = sigmoid(ctr_logit)
    cvr = sigmoid(cvr_logit)
    p, q, delta, u = solve_dual(ctr, cvr, wp, budget, target_cpc, epsilon)

    return bids(ctr, cvr, p, q, u, delta, target_cpc, epsilon)
