from experiment.non_robust_bid.lp import solve_dual
from experiment.non_robust_bid.bid import bids
from experiment.utils.utils import sigmoid


def non_robust_bid(
    config,
    ctr_logit,
    cvr_logit,
    wp,
    budget,
    target_cpc,
):
    ctr = sigmoid(ctr_logit)
    cvr = sigmoid(cvr_logit)
    p, q = solve_dual(config, ctr, cvr, wp, budget, target_cpc)

    return bids(ctr, cvr, p, q, target_cpc)
