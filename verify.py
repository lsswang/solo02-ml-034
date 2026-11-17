"""Self-checks for the position-aware click model and the joint allocator.

Run ``python3 verify.py``.  Every check prints PASS/FAIL; the process exits
non-zero on the first failure.  The script deliberately uses tiny,
hand-checkable instances and independent finite differences / brute-force
enumeration rather than relying on the production code paths to validate
themselves.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from starter.allocation import allocate
from starter.click_model import (
    examination,
    fit,
    objective,
    probabilities,
)


PASSED = 0


def check(name, condition, detail=""):
    global PASSED
    if not condition:
        raise AssertionError(f"FAIL: {name} {detail}")
    PASSED += 1
    print(f"PASS {name}")


def close(a, b, atol, rtol):
    return abs(a - b) <= atol + rtol * abs(b)


def numerical_gradient(theta, rows, ridge, positions, dimension, step=1e-6):
    # Clip the stencil to the legal box so the finite difference never asks
    # for an illegal parameter vector; the clipped points are still one-sided
    # valid samples.
    grad = np.empty(len(theta))
    for i in range(len(theta)):
        up = np.clip(theta.copy(), -3.0, 3.0)
        down = up.copy()
        up[i] = min(3.0, up[i] + step)
        down[i] = max(-3.0, down[i] - step)
        span = up[i] - down[i]
        if span == 0.0:
            grad[i] = 0.0
            continue
        grad[i] = (
            objective(up, rows, ridge, positions=positions, dimension=dimension)[0]
            - objective(down, rows, ridge, positions=positions, dimension=dimension)[0]
        ) / span
    return grad


def check_gradient(name, theta, rows, ridge, positions, dimension):
    value, analytic = objective(
        theta, rows, ridge, positions=positions, dimension=dimension
    )
    numeric = numerical_gradient(theta, rows, ridge, positions, dimension)
    bad = [
        (i, analytic[i], numeric[i])
        for i in range(len(theta))
        if not close(analytic[i], numeric[i], atol=2e-6, rtol=2e-5)
    ]
    check(
        name,
        not bad,
        detail=f"value={value!r} mismatches={bad}",
    )
    check(
        f"{name} (objective finite and gradient shape)",
        math.isfinite(value) and analytic.shape == theta.shape,
    )
    return analytic


def independent_objective(theta, rows, ridge, positions, dimension):
    """Straightforward loop implementation used only as a reference."""

    w = theta[:dimension]
    s = theta[dimension:]
    total = 0.0
    for row in rows:
        z = sum(row["features"][i] * w[i] for i in range(dimension))
        r = 1.0 / (1.0 + math.exp(-z))
        e = 1.0
        for k in range(1, row["position"] + 1):
            e *= 1.0 / (1.0 + math.exp(-s[k - 1]))
        p = e * r
        y = row["click"]
        total += row["weight"] * (-(y * math.log(p) + (1 - y) * math.log(1.0 - p)))
    total += ridge / 2.0 * (sum(v * v for v in w) + sum(v * v for v in s))
    return total


def test_objective_value_against_reference():
    rows = [
        {"features": [0.2, -0.1], "position": 0, "click": 1, "weight": 11},
        {"features": [0.2, -0.1], "position": 1, "click": 0, "weight": 9},
        {"features": [-0.7, 0.3], "position": 2, "click": 1, "weight": 5},
        {"features": [1.0, -1.0], "position": 2, "click": 0, "weight": 1},
    ]
    theta = np.array([0.8, -0.4, 0.5, -0.9])
    value, _ = objective(theta, rows, 0.2, positions=3, dimension=2)
    reference = independent_objective(theta, rows, 0.2, 3, 2)
    check(
        "objective value matches an independent computation",
        close(value, reference, atol=1e-8, rtol=1e-8),
        detail=f"{value} vs {reference}",
    )


def test_click_gradients():
    # P = 1: there are no examination parameters, and the position-0 click
    # probability is exactly sigmoid(x . w).
    rows_p1 = [
        {"features": [0.5], "position": 0, "click": 1, "weight": 3},
        {"features": [-0.5], "position": 0, "click": 0, "weight": 7},
        {"features": [1.0, -1.0], "position": 0, "click": 0, "weight": 2},
    ]
    # The first row set has 1 dimension, the second two, so separate calls.
    check_gradient(
        "P=1 gradient (D=1)",
        np.array([0.4]),
        rows_p1[:2],
        0.3,
        positions=1,
        dimension=1,
    )
    check_gradient(
        "P=1 gradient (D=2)",
        np.array([0.4, -1.2]),
        [rows_p1[2]],
        0.3,
        positions=1,
        dimension=2,
    )

    # P = 3 with repeated rows at every display position, including exact
    # duplicate records whose weights must both be used.  Every row uses the
    # same two-dimensional feature vector shape.
    rows_p3 = [
        {"features": [0.2, -0.1], "position": 0, "click": 1, "weight": 11},
        {"features": [0.2, -0.1], "position": 0, "click": 0, "weight": 4},
        {"features": [0.2, -0.1], "position": 1, "click": 1, "weight": 6},
        {"features": [0.2, -0.1], "position": 1, "click": 0, "weight": 9},
        {"features": [0.2, -0.1], "position": 1, "click": 0, "weight": 9},  # duplicate
        {"features": [-0.7, 0.3], "position": 2, "click": 1, "weight": 5},
        {"features": [-0.7, 0.3], "position": 2, "click": 0, "weight": 12},
        {"features": [1.0, -1.0], "position": 2, "click": 0, "weight": 1},
    ]
    theta = np.array([0.8, -0.4, 0.5, -0.9])
    analytic = check_gradient(
        "P=3 repeated rows gradient", theta, rows_p3, 0.2, positions=3, dimension=2
    )
    # The analytic gradient must have the w-then-s order.
    check(
        "gradient order is [w, s]",
        len(analytic) == 4,
    )

    # Duplicate-row weight preservation: doubling the duplicate's weight (9
    # -> 18) must change the objective by nine times that row's own loss.
    doubled = [dict(r) for r in rows_p3]
    doubled[4]["weight"] = 18
    v_original = objective(theta, rows_p3, 0.2, positions=3, dimension=2)[0]
    v_doubled = objective(theta, doubled, 0.2, positions=3, dimension=2)[0]
    w = theta[:2]
    p = examination(theta[2:], 1) * float(
        1.0 / (1.0 + math.exp(-float(np.dot([0.2, -0.1], w))))
    )
    row_loss = -math.log(1.0 - p)  # this duplicate row is an unclick
    check(
        "duplicate rows keep all weight",
        close(v_doubled - v_original, 9.0 * row_loss, atol=1e-8, rtol=1e-8),
    )

    # A large position-zero sigmoid is the cancellation-sensitive corner.
    check_gradient(
        "P=1 steep position-0 gradient",
        np.array([3.0]),
        [{"features": [1.0], "position": 0, "click": 1, "weight": 50}],
        0.01,
        positions=1,
        dimension=1,
        )
    # Near the lower box corner with a non-clicked row (kept far enough inside
    # the box that the finite-difference stencil stays legal).
    check_gradient(
        "near-box-corner gradient",
        np.array([-2.97, 2.97, -2.97, 2.97]),
        rows_p3,
        1.0,
        positions=3,
        dimension=2,
    )


def test_probability_semantics():
    rows = [
        {"features": [1.0], "position": 0, "click": 1, "weight": 5},
        {"features": [1.0], "position": 1, "click": 0, "weight": 5},
        {"features": [1.0], "position": 2, "click": 0, "weight": 5},
    ]
    model = fit(rows, dimension=1, positions=3, ridge=0.3)
    out = probabilities(model, [[0.5]], position=2)
    r = float(probabilities(model, [[0.5]], position=0)["r"][0])
    e = examination(model["s"], 2)
    check("p is e * r, not r", close(out["p"][0], e * r, atol=1e-12, rtol=1e-12))
    check("e is strictly less than one at deep positions", 0.0 < e < 1.0)
    both = probabilities(model, [[0.5]])
    check(
        "probabilities() returns r, e and p",
        set(both) == {"r", "e", "p"} and len(both["e"]) == 3,
    )
    all_positions = probabilities(model, [[0.5]])["p"][0]
    check(
        "full position axis matches per-position queries",
        np.allclose(
            all_positions,
            [float(probabilities(model, [[0.5]], position=j)["p"][0]) for j in range(3)],
        ),
    )


def test_stopping_and_box():
    rows = [
        {"features": [1.0, 0.0], "position": 0, "click": 1, "weight": 40},
        {"features": [1.0, 0.0], "position": 1, "click": 0, "weight": 40},
        {"features": [-1.0, 0.0], "position": 0, "click": 0, "weight": 20},
        {"features": [0.3, 0.4], "position": 2, "click": 1, "weight": 10},
    ]
    full = fit(rows, dimension=2, positions=3, ridge=0.1)
    check(
        "default fit converges on supplied training material",
        full["converged"] and full["status"] == "converged",
        detail=f"residual={full['projected_gradient_residual']}",
    )
    check(
        "fit objective is no higher than initial",
        full["objective"] <= full["initial_objective"] + 1e-12,
    )
    check(
        "converged residual <= 1e-5",
        full["projected_gradient_residual"] <= 1e-5,
    )
    check(
        "all returned parameters are inside the box",
        all(abs(v) <= 3.0 for v in full["theta"]),
    )
    # A one-iteration budget must be reported honestly, not as success.
    short = fit(rows, dimension=2, positions=3, ridge=0.1, max_iter=1)
    check(
        "iteration limit is reported and not called converged",
        short["status"] in {"max_iter", "not_converged"} and not short["converged"],
        detail=f"status={short['status']} residual={short['projected_gradient_residual']}",
    )
    check(
        "best legal model retained after early stop",
        all(abs(v) <= 3.0 for v in short["theta"])
        and short["objective"] <= short["initial_objective"] + 1e-12,
    )
    # Box-blocked optimum: with a strong click signal at the upper bound the
    # projected-gradient residual must use the projection, not the raw
    # gradient.
    blocked = fit(
        [{"features": [1.0], "position": 0, "click": 1, "weight": 50}],
        dimension=1,
        positions=1,
        ridge=0.01,
        initial={"w": [2.9]},
    )
    projected = np.clip(
        np.array(blocked["theta"]) - np.array(blocked["gradient"]), -3.0, 3.0
    )
    residual = np.max(np.abs(np.array(blocked["theta"]) - projected))
    check(
        "projected-gradient residual uses the box projection",
        close(residual, blocked["projected_gradient_residual"], 0, 1e-12),
    )
    check("box-blocked fit reports the correct status", blocked["converged"])


def brute_force_best(U, M, P, v, allowed, stock, category, mins, maxs):
    """Independent complete enumeration with the same legal semantics."""

    best = None
    used = None

    def user_options(u):
        options = []
        for perm in itertools.permutations([i for i in range(M) if allowed[u][i]], P):
            counts = [0] * len(mins[u])
            for item in perm:
                counts[category[item]] += 1
            if all(mins[u][c] <= counts[c] <= maxs[u][c] for c in range(len(counts))):
                options.append((sum(v[u, perm[j], j] for j in range(P)), perm))
        return options

    per_user = [user_options(u) for u in range(U)]
    state = {"best": None}

    def rec(u, used, total, assignment):
        if u == U:
            if state["best"] is None or total > state["best"] + 1e-9:
                state["best"] = total
            return
        for value, perm in per_user[u]:
            if any(used[i] >= stock[i] for i in perm):
                continue
            for i in perm:
                used[i] += 1
            assignment.append(perm)
            rec(u + 1, used, total + value, assignment)
            assignment.pop()
            for i in perm:
                used[i] -= 1

    rec(0, [0] * M, 0.0, [])
    return state["best"]


def verify_assignment(result, U, M, P, v, allowed, stock, category, mins, maxs):
    lists = result["assignment"]
    if lists is None or len(lists) != U:
        return False, "bad shape"
    used = [0] * M
    for u in range(U):
        if len(set(lists[u])) != P:
            return False, f"user {u} repeats an item"
        counts = [0] * len(mins[u])
        for j in range(P):
            item = lists[u][j]
            if not (0 <= item < M) or not allowed[u][item]:
                return False, f"user {u} uses a disallowed item"
            used[item] += 1
            counts[category[item]] += 1
        for c in range(len(counts)):
            if not (mins[u][c] <= counts[c] <= maxs[u][c]):
                return False, f"user {u} category {c} limits"
    for i in range(M):
        if used[i] > stock[i]:
            return False, f"item {i} oversold"
    value = sum(v[u, lists[u][j], j] for u in range(U) for j in range(P))
    return True, value


def test_allocation():
    rng = np.random.default_rng(12345)
    checked = 0
    for trial in range(120):
        U = int(rng.integers(1, 5))
        M = int(rng.integers(1, 7))
        P = int(rng.integers(1, 4))
        v = rng.random((U, M, P))
        allowed = rng.random((U, M)) < 0.8
        stock = rng.integers(1, U + 1, size=M)
        category = rng.integers(0, 3, size=M)
        C = int(max(category)) + 1
        mins = []
        maxs = []
        for u in range(U):
            lo = []
            hi = []
            for _c in range(C):
                # Keep most lower bounds at zero so that random instances are
                # usually feasible; still exercise non-trivial bounds.
                a = int(rng.integers(0, P + 1)) if rng.random() < 0.3 else 0
                b = int(rng.integers(a, P + 1))
                lo.append(a)
                hi.append(b)
            mins.append(lo)
            maxs.append(hi)

        result = allocate(v, allowed, stock, category, mins, maxs, method="milp")
        brute = brute_force_best(U, M, P, v, allowed, stock, category, mins, maxs)
        if brute is None:
            check(
                f"infeasibility agrees with brute force (trial {trial})",
                result["status"] == "infeasible",
                detail=f"trial={trial} status={result['status']}",
            )
            continue
        check(
            f"MILP finds the brute-force optimum (trial {trial})",
            result["status"] == "optimal",
            detail=f"status={result['status']}",
        )
        ok, recomputed = verify_assignment(
            result, U, M, P, v, allowed, stock, category, mins, maxs
        )
        check(f"reported list is legal (trial {trial})", ok, detail=str(recomputed))
        check(
            f"reported value is recomputed from the list (trial {trial})",
            close(result["value"], recomputed, atol=1e-8, rtol=1e-9)
            and close(recomputed, brute, atol=1e-8, rtol=1e-9),
            detail=f"reported={result['value']} recomputed={recomputed} brute={brute}",
        )
        checked += 1
    check("allocation cross-check covered several feasible cases", checked >= 5)

    # P > M can never be feasible without repeating an item.
    v = rng.random((2, 2, 3))
    result = allocate(
        v, [[True, True], [True, True]], [2, 2], [0, 0], [[0, 0], [0, 0]], [[3, 3], [3, 3]]
    )
    check("P > M reports infeasible rather than guessing", result["status"] == "infeasible")

    # Hand instance: equal allowed items, scarce stock, category lower bound.
    v = np.array(
        [
            [[0.9, 0.1], [0.8, 0.7], [0.2, 0.3]],
            [[0.9, 0.1], [0.1, 0.2], [0.8, 0.9]],
        ]
    )
    allowed = [[True, True, True], [True, True, True]]
    # Four slots are needed in total; only item 2 has enough stock for both
    # users, while items 0 and 1 are genuinely scarce.
    stock = [1, 1, 2]
    category = [0, 0, 1]
    mins = [[0, 0], [0, 1]]
    maxs = [[2, 2], [2, 2]]
    result = allocate(v, allowed, stock, category, mins, maxs)
    check(
        "scarce shared item is not oversold",
        result["status"] == "optimal"
        and [usage["used"] for usage in result["stock_usage"]] == [1, 1, 2],
    )
    # Item 0 is best for both users in slot 0; only one may have it.
    check(
        "joint solution assigns the scarce favourite to one user only",
        sum(item == 0 for u in range(2) for item in result["assignment"][u]) == 1,
    )

    # Independent greedy failure demonstration.  Item 2 is the only category-1
    # item and has stock one.  User 0 does not need category 1 but greedily
    # takes item 2 in slot 0 because it is its favourite.  User 1 then cannot
    # satisfy its category-1 lower bound, even though a joint solution (user 1
    # gets item 2, user 0 takes the category-0 items) is feasible.
    v = np.array(
        [
            [[0.4, 0.5], [0.3, 0.2], [0.99, 0.0]],  # user 0, items 0..2
            [[0.1, 0.1], [0.1, 0.1], [0.9, 0.0]],  # user 1, items 0..2
        ]
    )
    allowed = [[True, True, True], [True, True, True]]
    stock = [2, 2, 1]
    category = [0, 0, 1]
    mins = [[0, 0], [0, 1]]
    maxs = [[2, 2], [2, 2]]
    joint = allocate(v, allowed, stock, category, mins, maxs)
    greedy = [[2, 0], [0, 1]]
    ok, _ = verify_assignment(
        {"assignment": greedy}, 2, 3, 2, v, allowed, stock, category, mins, maxs
    )
    check("naive fixed-order greedy list violates legal constraints", not ok)
    check(
        "joint optimal list is feasible where greedy is not",
        joint["status"] == "optimal",
        detail=f"status={joint['status']}",
    )
    ok_joint, _ = verify_assignment(
        joint, 2, 3, 2, v, allowed, stock, category, mins, maxs
    )
    check("joint list is legal in the greedy-failure instance", ok_joint)


def test_external_tensor_is_independent():
    # No log, no fitted model: just a legal U x M x P tensor.
    v = np.array([[[0.1, 0.9], [0.8, 0.2], [0.3, 0.3]]])
    result = allocate(v, [[True, True, True]], [1, 1, 1], [0, 0, 1], [[0, 0]], [[2, 2]])
    check(
        "allocation accepts a stand-alone value tensor",
        result["status"] == "optimal" and result["assignment"] == [[1, 0]],
    )


def main():
    test_objective_value_against_reference()
    test_click_gradients()
    test_probability_semantics()
    test_stopping_and_box()
    test_allocation()
    test_external_tensor_is_independent()
    print(f"\nAll {PASSED} checks passed.")


if __name__ == "__main__":
    main()
