import numpy as np


def bids(ctr, cvr, p, q, u, delta, target_cpc, epsilon):
    alpha = np.sqrt(2 * epsilon)

    bids = 1 / (p + q) * (
        - cvr * delta - alpha * target_cpc * u + target_cpc * q * ctr
    )

    return bids
