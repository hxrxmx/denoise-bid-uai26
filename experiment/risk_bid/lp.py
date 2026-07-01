import cvxpy as cp
import numpy as np


def solve_dual(config, ctr, cvr, wp, budget, target_cpc, sigma_logit_ctr, alpha=0.1):
    num_auctions = len(ctr)
    std_ctr = sigma_logit_ctr * (ctr * (1 - ctr))

    ctr_adj = np.maximum(ctr - alpha * std_ctr, 1e-10)

    p = cp.Variable(nonneg=True)
    r = cp.Variable(num_auctions, nonneg=True)

    objective = p * budget + cp.sum(r)

    constraints = [
        p * wp + r >= cvr * ctr_adj
    ]

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(
        solver=cp.MOSEK,
        mosek_params={
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS": config.mosek.tol_pfeas,
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS": config.mosek.tol_dfeas,
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": config.mosek.tol_rel_gap,
            "MSK_DPAR_INTPNT_CO_TOL_MU_RED": config.mosek.tol_mu_red,
        }
    )

    return p.value if p.value is not None else 1.0


