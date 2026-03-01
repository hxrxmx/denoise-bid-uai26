import numpy as np
from astroML.density_estimation import XDGMM


def fit_gmm(
    ctr_logits,
    cvr_logits,
    ctr_logit_sigmas,
    cvr_logit_sigmas,
    n_components,
):
    ctr_logit_variances, cvr_logit_variances = \
        ctr_logit_sigmas**2, cvr_logit_sigmas**2
    X = np.column_stack([ctr_logits, cvr_logits])

    N = X.shape[0]
    Xerr = np.zeros((N, 2, 2))
    Xerr[:, 0, 0] = ctr_logit_variances.clip(1e-12, np.inf)
    Xerr[:, 1, 1] = cvr_logit_variances.clip(1e-12, np.inf)

    xdgmm = XDGMM(n_components=n_components, max_iter=300, tol=1e-4)
    xdgmm.fit(X, Xerr)

    return xdgmm.alpha, xdgmm.mu, xdgmm.V
