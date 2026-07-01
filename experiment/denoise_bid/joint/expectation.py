import numpy as np
from scipy.special import expit, logsumexp
from experiment.utils.utils import sigmoid


def ctr_expectation(
    ctr_logits,
    cvr_logits,
    sigma_ctr,
    sigma_cvr,
    weights,
    means,
    sigmas,
):
    zeta_hat = np.stack([ctr_logits, cvr_logits], axis=1)
    theta_00 = sigma_ctr**2
    theta_11 = sigma_cvr**2

    numerator = np.zeros_like(ctr_logits)
    denominator = np.zeros_like(ctr_logits)

    for k in range(len(weights)):
        mu = means[k]
        sigma = sigmas[k]
        weight = weights[k]

        s_00 = sigma[0, 0] + theta_00
        s_01 = sigma[0, 1]
        s_10 = sigma[1, 0]
        s_11 = sigma[1, 1] + theta_11

        det_s = s_00 * s_11 - s_01 * s_10
        inv_s_00 = s_11 / det_s
        inv_s_01 = -s_01 / det_s
        inv_s_10 = -s_10 / det_s
        inv_s_11 = s_00 / det_s

        diff = mu - zeta_hat
        quad_form = (
            diff[:, 0]**2 * inv_s_00 +
            diff[:, 0] * diff[:, 1] * (inv_s_01 + inv_s_10) +
            diff[:, 1]**2 * inv_s_11
        )

        det_sigma = sigma[0, 0] * sigma[1, 1] - sigma[0, 1] * sigma[1, 0]
        inv_sigma_00 = sigma[1, 1] / det_sigma
        inv_sigma_01 = -sigma[0, 1] / det_sigma
        inv_sigma_10 = -sigma[1, 0] / det_sigma
        inv_sigma_11 = sigma[0, 0] / det_sigma

        inv_theta_00 = 1.0 / theta_00
        inv_theta_11 = 1.0 / theta_11

        inv_sigma_prime_00 = inv_sigma_00 + inv_theta_00
        inv_sigma_prime_01 = inv_sigma_01
        inv_sigma_prime_10 = inv_sigma_10
        inv_sigma_prime_11 = inv_sigma_11 + inv_theta_11

        det_inv_sigma_prime = (
            inv_sigma_prime_00 * inv_sigma_prime_11 -
            inv_sigma_prime_01 * inv_sigma_prime_10
        )

        v_0 = inv_sigma_00 * mu[0] + inv_sigma_01 * mu[1] + \
            inv_theta_00 * zeta_hat[:, 0]
        v_1 = inv_sigma_10 * mu[0] + inv_sigma_11 * mu[1] + \
            inv_theta_11 * zeta_hat[:, 1]

        mu_prime_x = (inv_sigma_prime_11 / det_inv_sigma_prime) * v_0 + \
            (-inv_sigma_prime_01 / det_inv_sigma_prime) * v_1
        sigma_prime_x_var = inv_sigma_prime_11 / det_inv_sigma_prime

        weight_prime = (weight / det_s) * np.exp(-0.5 * quad_form)

        expectation_sigmoid = sigmoid(
            mu_prime_x / np.sqrt(1 + (np.pi / 8.0) * sigma_prime_x_var)
        )

        numerator += weight_prime * expectation_sigmoid
        denominator += weight_prime

    return numerator / denominator


def value_expectation(
    ctr_logit,
    cvr_logit,
    ctr_sigma,
    cvr_sigma,
    weights,
    means,
    sigmas,
    n_quad_points=5,
    herm_gaus_points=None,
):
    ctr_logit_var = ctr_sigma**2
    cvr_logit_var = cvr_sigma**2

    Y = np.column_stack([ctr_logit, cvr_logit])[:, np.newaxis, :]

    D = np.zeros((Y.shape[0], 1, 2, 2))
    D[:, 0, 0, 0] = ctr_logit_var
    D[:, 0, 1, 1] = cvr_logit_var

    mu_prior = means[np.newaxis, :, :]
    Sigma_prior = sigmas[np.newaxis, :, :, :]

    T = Sigma_prior + D

    try:
        T_inv = np.linalg.inv(T)
    except np.linalg.LinAlgError:
        T_inv = np.linalg.inv(T + np.eye(2) * 1e-6)

    delta = Y - mu_prior

    mahalanobis = np.einsum('nki,nkij,nkj->nk', delta, T_inv, delta)
    _, log_det_T = np.linalg.slogdet(T)

    log_lik = -0.5 * (2 * np.log(2 * np.pi) + log_det_T + mahalanobis)

    log_post_weights_unnorm = np.log(weights + 1e-300)[np.newaxis, :] + log_lik

    log_evidence = logsumexp(log_post_weights_unnorm, axis=1, keepdims=True)
    post_weights = np.exp(log_post_weights_unnorm - log_evidence)

    W = np.einsum('nkij,nkjm->nkim', Sigma_prior, T_inv)

    mu_post = mu_prior + np.einsum('nkij,nkj->nki', W, delta)

    Sigma_post = Sigma_prior - np.einsum('nkij,nkjl->nkil', W, Sigma_prior)

    base_points, base_weights = herm_gaus_points if herm_gaus_points is not None else np.polynomial.hermite.hermgauss(n_quad_points)

    grid_x, grid_y = np.meshgrid(base_points, base_points)
    grid_points_flat = np.vstack([grid_x.ravel(), grid_y.ravel()])
    grid_weights_flat = (np.outer(base_weights, base_weights)).ravel()

    # n_total_points = grid_points_flat.shape[1]

    L = np.linalg.cholesky(
        Sigma_post + np.eye(2)[np.newaxis, np.newaxis, :, :] * 1e-8
    )

    mu_exp = mu_post[:, :, :, np.newaxis]
    L_exp = L
    pts_exp = grid_points_flat[np.newaxis, np.newaxis, :, :]

    transformed_points = mu_exp + np.sqrt(2) * np.matmul(L_exp, pts_exp)

    val_ctr = expit(transformed_points[:, :, 0, :])
    val_cvr = expit(transformed_points[:, :, 1, :])

    func_values = val_ctr * val_cvr

    integral_approx = (1 / np.pi) * np.sum(
        func_values * grid_weights_flat,
        axis=2
    )

    final_expectation = np.sum(post_weights * integral_approx, axis=1)

    return final_expectation
