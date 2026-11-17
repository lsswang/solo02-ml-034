"""Synthetic experiments: estimate a position-aware click model, build the
shared value tensor, then allocate one joint list batch.

Run with ``python3 -m starter.experiments`` (or ``python3 experiments.py``).
The output is deterministic JSON-friendly data plus short human-readable
summary lines.

The scientific point:

1. Treating every unclicked impression as "dislike" is wrong when clicks are
   only observed after examination.  Fitting ``p = e[j] * r(x)`` separates
   relevance from the position cascade.
2. Optimising each user independently, then deduplicating shared stock
   afterwards, either oversells items or gives away value.  The joint MILP
   allocates all users at once.
3. Rows that look like the same feature vector but come from different
   positions carry different information.  Ignoring position produces a
   relevance estimate contaminated by the examination cascade.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.special import expit

from .allocation import allocate
from .click_model import fit, probabilities

RNG_SEED = 20240601


def _simulate_rows(w_true, s_true, positions, n, seed):
    """Generate weighted display rows for one evidence set."""

    rng = np.random.default_rng(seed)
    D = len(w_true)
    rows = []
    for _ in range(n):
        x = rng.uniform(-1.0, 1.0, size=D)
        r = float(expit(x @ w_true))
        j = int(rng.integers(0, positions))
        e = 1.0
        for k in range(1, j + 1):
            e *= float(expit(s_true[k - 1]))
        click = int(rng.random() < e * r)
        weight = int(rng.integers(1, 26))
        rows.append(
            {
                "features": [float(value) for value in x],
                "position": j,
                "click": click,
                "weight": weight,
            }
        )
    return rows


def _naive_relevance(rows, dimension, ridge=0.1):
    """Relevance-only logistic fit on clicks, which ignores position."""

    X = np.array([row["features"] for row in rows], dtype=float)
    y = np.array([row["click"] for row in rows], dtype=float)
    weight = np.array([row["weight"] for row in rows], dtype=float)

    def fun(theta):
        z = X @ theta
        loss = float(
            np.sum(weight * (np.logaddexp(0.0, z) - y * z))
            + ridge / 2.0 * np.dot(theta, theta)
        )
        grad = X.T @ (weight * (expit(z) - y)) + ridge * theta
        return loss, grad

    from scipy.optimize import minimize

    result = minimize(
        fun,
        np.zeros(dimension),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-3.0, 3.0)] * dimension,
    )
    return result.x


def _value_tensor(W, s, features, positions):
    """v[u, i, j] = e[j] * sigmoid(features[u, i] . w)."""

    w = np.asarray(W, dtype=float)
    s = np.asarray(s, dtype=float)
    x = np.asarray(features, dtype=float)
    U, M, D = x.shape
    e = np.ones(positions, dtype=float)
    for j in range(1, positions):
        e[j] = e[j - 1] * float(expit(s[j - 1]))
    r = expit(x @ w)  # U x M
    v = r[:, :, None] * e[None, None, :]
    return v, r, e


def _greedy_shared(v, allowed, stock, positions):
    """Fixed-user-order greedy that ignores stock until it runs out.

    This is the failure mode the joint allocator must beat: take each user's
    individually best list, decrement shared stock, and then patch with
    whatever is left.
    """

    U, M, P = v.shape
    used = [0] * M
    lists = []
    for u in range(U):
        chosen = []
        for j in range(P):
            order = sorted(
                (i for i in range(M) if allowed[u][i] and i not in chosen),
                key=lambda i: (-v[u, i, j], i),
            )
            pick = None
            for i in order:
                if used[i] < stock[i]:
                    pick = i
                    break
            if pick is None:
                return None  # greedy cannot produce a complete list
            chosen.append(pick)
            used[pick] += 1
        lists.append(chosen)
    return lists


def _recompute(lists, v):
    if lists is None:
        return None
    return float(
        sum(v[u, lists[u][j], j] for u in range(len(lists)) for j in range(len(lists[u])))
    )


def run():
    report = {}

    # ---- 1. Learn the position-aware model from one evidence set ----------
    w_true = np.array([1.1, -0.8])
    s_true = np.array([-1.2, -0.9])  # deep positions are rarely examined
    positions = 3
    rows = _simulate_rows(w_true, s_true, positions, n=200, seed=5)

    model = fit(
        rows,
        dimension=2,
        positions=positions,
        ridge=0.1,
        initial={"w": [0.0, 0.0], "s": [0.0, 0.0]},
    )
    naive_w = _naive_relevance(rows, dimension=2)
    report["learning"] = {
        "truth": {"w": w_true.tolist(), "s": s_true.tolist()},
        "position_aware": {
            "w": model["w"],
            "s": model["s"],
            "objective": model["objective"],
            "initial_objective": model["initial_objective"],
            "projected_gradient_residual": model["projected_gradient_residual"],
            "converged": model["converged"],
            "status": model["status"],
        },
        "naive_relevance_only_w": naive_w.tolist(),
    }

    # ---- 2. Build the shared value tensor and allocate --------------------
    # Six users, eight items, three slots.  Features are fixed across the
    # evidence change in part 3.
    U, M, P = 6, 8, 3
    rng = np.random.default_rng(RNG_SEED + 1)
    features = rng.uniform(-1.0, 1.0, size=(U, M, 2))
    allowed = np.ones((U, M), dtype=bool)
    # Item 1 is scarce, so two users would both like it but only one may have
    # it; items 5, 6 and 7 are the only source of category 2, which every user
    # needs at least once.
    stock = [2, 1, 3, 3, 3, 2, 2, 2]
    category = [0, 0, 0, 1, 1, 2, 2, 2]
    C = 3
    category_min = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]
    category_max = [[2, 2, 1], [2, 2, 1], [2, 2, 1], [2, 2, 1], [2, 2, 1], [2, 2, 1]]

    def allocate_for(model):
        v, r, e = _value_tensor(model["w"], model["s"], features, P)
        joint = allocate(
            v,
            allowed,
            stock,
            category,
            category_min,
            category_max,
            method="milp",
        )
        return v, r, e, joint

    v1, r1, e1, joint1 = allocate_for(model)
    independent = []
    for u in range(U):
        order = sorted(range(M), key=lambda i: (-v1[u, i, 0], i))[:P]
        independent.append(order)
    greedy = _greedy_shared(v1, allowed, stock, P)

    report["allocation"] = {
        "value_tensor": {
            "shape": list(v1.shape),
            "values": v1.tolist(),
        },
        "relevance": r1.tolist(),
        "examination": e1.tolist(),
        "model": {"w": model["w"], "s": model["s"]},
        "features": features.tolist(),
        "allowed": allowed.tolist(),
        "category": category,
        "category_min": category_min,
        "category_max": category_max,
        "stock": stock,
        "joint": {
            "status": joint1["status"],
            "lists": joint1["assignment"],
            "value": joint1["value"],
            "stock_usage": joint1["stock_usage"],
            "category_counts": joint1["category_counts"],
        },
        "independent_lists": independent,
        "independent_value_if_ignoring_stock": _recompute(independent, v1),
        "fixed_order_greedy": {
            "lists": greedy,
            "value": _recompute(greedy, v1),
        },
    }

    # ---- 3. Change only the training evidence, refit, reallocate ---------
    # This evidence has a much shallower cascade (s closer to 0), so position
    # one is nearly as visible as position zero.  Features, allowed, stock and
    # category limits stay identical.
    rows2 = _simulate_rows(w_true, np.array([0.6, 0.6]), positions, n=140, seed=RNG_SEED + 7)
    model2 = fit(
        rows2,
        dimension=2,
        positions=positions,
        ridge=0.1,
        initial={"w": [0.0, 0.0], "s": [0.0, 0.0]},
    )
    v2, r2, e2, joint2 = allocate_for(model2)

    report["refit"] = {
        "model": {
            "w": model2["w"],
            "s": model2["s"],
            "objective": model2["objective"],
            "projected_gradient_residual": model2["projected_gradient_residual"],
            "converged": model2["converged"],
            "status": model2["status"],
        },
        "examination": e2.tolist(),
        "value_tensor": {"shape": list(v2.shape), "values": v2.tolist()},
        "joint": {
            "status": joint2["status"],
            "lists": joint2["assignment"],
            "value": joint2["value"],
            "stock_usage": joint2["stock_usage"],
        },
        "changed_lists": joint2["assignment"] != joint1["assignment"],
    }
    return report


def main():
    report = run()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
