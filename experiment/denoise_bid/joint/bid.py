def bids(ctr, value, p, q, target_cpc):
    bids = 1 / (p + q) * value + q * target_cpc / (p + q) * ctr
    return bids
