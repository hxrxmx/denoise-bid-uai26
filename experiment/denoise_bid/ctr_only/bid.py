def bids(ctr, cvr, p, q, target_cpc):
    bids = ctr / (p + q) * (cvr + q * target_cpc)
    return bids
