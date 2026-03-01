import cvxpy as cp
import numpy as np


def solve_dual(ctr, cvr, wp, budget, target_cpc, epsilon):
    alpha = np.sqrt(2 * epsilon)
    num_auctions = len(ctr)

    delta = cp.Variable(num_auctions)
    u = cp.Variable(num_auctions)
    gamma = cp.Variable(nonneg=True)
    u0 = cp.Variable(nonneg=True)
    s = cp.Variable(num_auctions)

    expr = (
        -cp.multiply(delta, cvr)
        - gamma * wp
        - target_cpc * alpha * u
        - u0 * wp
        + target_cpc * u0 * ctr
    )

    constraints = [
        cp.norm(delta + ctr, 2) <= alpha,
        cp.norm(u, 2) <= u0,
        s >= expr,
    ]

    objective = gamma * budget + cp.sum(cp.maximum(0, s))
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(
        solver=cp.MOSEK,
        mosek_params={
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-6,
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-6,
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-6,
            "MSK_DPAR_INTPNT_CO_TOL_MU_RED": 1e-6,
        }
    )

    delta_opt = delta.value
    u_opt = u.value
    gamma_opt = gamma.value
    u0_opt = u0.value

    return gamma_opt, u0_opt, delta_opt, u_opt
