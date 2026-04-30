import cvxpy as cp


def solve_dual(config, ctr, value, wp, budget, target_cpc):
    num_auctions = len(ctr)

    p = cp.Variable(nonneg=True)
    q = cp.Variable(nonneg=True)
    r = cp.Variable(num_auctions, nonneg=True)

    objective = p * budget + cp.sum(r)

    constraints = [
        wp * p + (wp - target_cpc * ctr) * q + r >= value,
    ]

    problem = cp.Problem(
        objective=cp.Minimize(objective),
        constraints=constraints,
    )

    problem.solve(
        solver=cp.MOSEK,
        mosek_params={
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS": config.mosek.tol_pfeas,
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS": config.mosek.tol_dfeas,
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": config.mosek.tol_rel_gap,
            "MSK_DPAR_INTPNT_CO_TOL_MU_RED": config.mosek.tol_mu_red,
        }
    )

    return (
        p.value if p.value is not None else 0.,
        q.value if q.value is not None else 0.,
    )
