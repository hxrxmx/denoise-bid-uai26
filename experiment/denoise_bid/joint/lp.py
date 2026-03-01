import cvxpy as cp


def solve_dual(ctr, value, wp, budget, target_cpc):
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
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-8,
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-8,
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-8,
            "MSK_DPAR_INTPNT_CO_TOL_MU_RED": 1e-8,
        }
    )

    return (
        p.value if p.value is not None else 0.,
        q.value if q.value is not None else 0.,
    )
