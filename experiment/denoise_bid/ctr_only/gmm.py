import numpy as np
from astroML.density_estimation import XDGMM


def fit_gmm(ctr_logit, ctr_sigma, n_components=3):
    N = ctr_logit.shape[0]

    ctr_gmm_logit = ctr_logit[:, np.newaxis]
    ctr_gmm_logit_corr = np.zeros((N, 1, 1))
    ctr_gmm_logit_corr[:, 0, 0] = ctr_sigma**2

    xdgmm = XDGMM(n_components=n_components, max_iter=300, tol=1e-4)
    xdgmm.fit(ctr_gmm_logit, ctr_gmm_logit_corr)

    return (
        xdgmm.alpha,
        xdgmm.mu.reshape(-1),
        np.sqrt(xdgmm.V.reshape(-1)),
    )
