"""Joint slot allocation with shared stock and per-user category limits.

The entry point :func:`allocate` consumes a user x item x position value
tensor.  Every value must lie in [0, 1].  Each user receives exactly one
allowed item in every position, no user may use the same item twice, item
``i`` can be used at most ``stock[i]`` times across all users, and each user
must respect per-category lower and upper bounds over the positions actually
filled.

The assignment is solved as a single MILP over all users, so users never
oversell shared stock and no fixed user order is used.  A complete
depth-first enumeration is available as an independent exact solver for small
instances and for cross-checking.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import itertools
import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

MIN_VALUE = 0.0
MAX_VALUE = 1.0
MAX_USERS = 6
MAX_ITEMS = 8
MAX_POSITIONS = 3
TOL = 1e-9


class Result(dict):
    """A dict whose entries are also readable as attributes."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - normal attribute miss
            raise AttributeError(name) from exc


class InputError(ValueError):
    """Raised for illegal shapes, indices, values or category limits."""


def _fail(message: str) -> None:
    raise InputError(message)


def _is_int(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (int, np.integer)):
        return True
    return isinstance(value, (float, np.floating)) and math.isfinite(float(value)) and float(
        value
    ).is_integer()


def _as_int(value, name: str) -> int:
    if not _is_int(value):
        _fail(f"{name} must be an integer, got {value!r}")
    return int(value)


def _as_int_list(value, name: str, length, low, high) -> list[int]:
    if isinstance(value, np.ndarray):
        values = value.tolist()
    elif isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(f"{name} must be a sequence of integers")
    else:
        values = list(value)
    if len(values) != length:
        _fail(f"{name} must have length {length}, got {len(values)}")
    result = []
    for index, item in enumerate(values):
        integer = _as_int(item, f"{name}[{index}]")
        if integer < low or integer > high:
            _fail(f"{name}[{index}] must be in {low}..{high}, got {integer}")
        result.append(integer)
    return result


def _as_matrix(value, name: str, shape, low, high) -> np.ndarray:
    try:
        matrix = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{name} must be a numeric array") from exc
    if matrix.shape != shape:
        _fail(f"{name} must have shape {shape}, got {matrix.shape}")
    try:
        numeric = matrix.astype(float)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{name} must contain real numbers") from exc
    if not np.all(np.isfinite(numeric)):
        _fail(f"{name} contains a non-finite value")
    if np.any(numeric < low) or np.any(numeric > high):
        _fail(f"every entry of {name} must be in [{low}, {high}]")
    return numeric


def _parse_instance(values, allowed, stock, category, category_min, category_max):
    if values is None:
        _fail("values is required")
    try:
        raw = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InputError("values must be a numeric U x M x P array") from exc
    if raw.ndim != 3:
        _fail(f"values must be a three-dimensional U x M x P array, got {raw.ndim} dimensions")
    U, M, P = raw.shape
    if not (1 <= U <= MAX_USERS):
        _fail(f"U must be in 1..{MAX_USERS}, got {U}")
    if not (1 <= M <= MAX_ITEMS):
        _fail(f"M must be in 1..{MAX_ITEMS}, got {M}")
    if not (1 <= P <= MAX_POSITIONS):
        _fail(f"P must be in 1..{MAX_POSITIONS}, got {P}")
    if not np.all(np.isfinite(raw)):
        _fail("values contains a non-finite entry")
    v = _as_matrix(raw, "values", (U, M, P), MIN_VALUE, MAX_VALUE)

    allowed_matrix = _as_matrix(allowed, "allowed", (U, M), 0, 1)
    # Reject non-boolean-looking allowed entries (for example 0.5).
    raw_allowed = np.asarray(allowed)
    if raw_allowed.dtype != bool:
        if not np.all((raw_allowed == 0) | (raw_allowed == 1)):
            _fail("allowed must contain booleans (or 0/1)")
    allowed_matrix = allowed_matrix.astype(bool)

    stock_list = _as_int_list(stock, "stock", M, 0, U)

    category_list = _as_int_list(category, "category", M, 0, 2)
    C = max(category_list) + 1 if category_list else 0

    mins, C = _parse_category_limits(category_min, "category_min", U, P, category_list, C)
    maxs, C_max = _parse_category_limits(
        category_max, "category_max", U, P, category_list, C
    )
    if C_max != C:
        _fail("category_min and category_max must have the same width")
    for u in range(U):
        for c in range(C):
            if mins[u][c] > maxs[u][c]:
                _fail(
                    f"category_min[{u}][{c}]={mins[u][c]} exceeds "
                    f"category_max[{u}][{c}]={maxs[u][c]}"
                )
    return v, allowed_matrix, stock_list, category_list, mins, maxs, U, M, P, C


def _parse_category_limits(
    value, name: str, U: int, P: int, category: list[int], num_categories: int
) -> tuple[list[list[int]], int]:
    """Parse a U x C category limit table.

    ``C`` is normally ``max(category)+1`` but a table with exactly three
    columns is also accepted when every item uses categories 0..1 (real
    callers often pass a fixed three-category table).  The returned ``C`` is
    the accepted width.
    """

    if isinstance(value, np.ndarray):
        if value.ndim != 2:
            _fail(f"{name} must be a U x C table")
        rows = value.tolist()
        raw_shape = value.shape
    elif isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(f"{name} must be a U x C table of integers")
    else:
        rows = list(value)
        raw_shape = None

    if len(rows) != U:
        _fail(f"{name} must have U={U} rows, got {len(rows)}")

    widths = set()
    for u, row in enumerate(rows):
        if isinstance(row, np.ndarray):
            row = row.tolist()
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            _fail(f"{name}[{u}] must be a sequence")
        widths.add(len(row))
    if len(widths) != 1:
        _fail(f"{name} rows have inconsistent widths {sorted(widths)}")
    width = widths.pop()
    if width < num_categories:
        _fail(
            f"{name} must have at least C={num_categories} columns for the "
            f"categories that occur in category, got {width}"
        )
    if width > 3:
        _fail(f"{name} may have at most 3 columns, got {width}")
    if width > num_categories:
        # A wider table is accepted as long as every extra column refers to a
        # category that no item uses.  Its bounds are still enforced by the
        # MILP, and because no item belongs to it the only feasible values
        # are zero, so such columns are harmless.
        num_categories = width

    result = []
    for u, row in enumerate(rows):
        if isinstance(row, np.ndarray):
            row = row.tolist()
        values = list(row)
        parsed = []
        for c, item in enumerate(values):
            integer = _as_int(item, f"{name}[{u}][{c}]")
            if integer < 0 or integer > P:
                _fail(f"{name}[{u}][{c}] must be in 0..{P}, got {integer}")
            parsed.append(integer)
        result.append(parsed)
    return result, num_categories


def _objective(assignment, v):
    return float(sum(v[u, assignment[u][j], j] for u in range(len(assignment)) for j in range(len(assignment[u]))))


def _feasible(assignment, allowed, stock, category, mins, maxs):
    U = len(assignment)
    P = len(assignment[0])
    used = {}
    for u in range(U):
        seen = set()
        counts = {}
        for j in range(P):
            item = assignment[u][j]
            if item < 0 or item >= allowed.shape[1]:
                return False
            if not allowed[u][item]:
                return False
            if item in seen:
                return False
            seen.add(item)
            used[item] = used.get(item, 0) + 1
            counts[category[item]] = counts.get(category[item], 0) + 1
        for c in range(len(mins[u])):
            count = counts.get(c, 0)
            if count < mins[u][c] or count > maxs[u][c]:
                return False
    for item, count in used.items():
        if count > stock[item]:
            return False
    return True


def _full_enumeration(v, allowed, stock, category, mins, maxs):
    """Exact enumeration of every complete legal list for all users.

    Returns ``(best_assignment, best_value)`` or ``(None, None)`` when no
    complete legal allocation exists.
    """

    U, M, P = v.shape
    C = len(mins[0])
    per_user = []
    for u in range(U):
        allowed_items = [i for i in range(M) if allowed[u][i]]
        options = []
        for perm in itertools.permutations(allowed_items, P):
            counts = [0] * C
            for item in perm:
                counts[category[item]] += 1
            if any(counts[c] < mins[u][c] or counts[c] > maxs[u][c] for c in range(C)):
                continue
            value = sum(v[u, perm[j], j] for j in range(P))
            options.append((value, tuple(perm)))
        per_user.append(options)

    if any(len(options) == 0 for options in per_user):
        return None, None

    best_value = None
    best_assignment = None
    remaining = [0] * M

    def search(u, used, value, assignment):
        nonlocal best_value, best_assignment
        if u == U:
            if best_value is None or value > best_value + TOL:
                best_value = value
                best_assignment = [list(items) for items in assignment]
            return
        for slot_value, perm in per_user[u]:
            ok = True
            for item in perm:
                if used[item] >= stock[item]:
                    ok = False
                    break
            if not ok:
                continue
            for item in perm:
                used[item] += 1
            assignment.append(perm)
            search(u + 1, used, value + slot_value, assignment)
            assignment.pop()
            for item in perm:
                used[item] -= 1

    search(0, remaining, 0.0, [])
    return best_assignment, best_value


def _solve_milp(v, allowed, stock, category, mins, maxs):
    U, M, P = v.shape
    C = len(mins[0])
    indices = [(u, i, j) for u in range(U) for i in range(M) for j in range(P)]
    n = len(indices)
    index_of = {key: k for k, key in enumerate(indices)}
    rows = []
    cols = []
    data = []
    lower = []
    upper = []
    row = 0

    def add(coefficients, lo, hi):
        nonlocal row
        for column, coefficient in coefficients:
            rows.append(row)
            cols.append(column)
            data.append(coefficient)
        lower.append(lo)
        upper.append(hi)
        row += 1

    # Exactly one item per user and position.
    for u in range(U):
        for j in range(P):
            add([(index_of[(u, i, j)], 1.0) for i in range(M)], 1.0, 1.0)

    # No repeated item within a user.
    for u in range(U):
        for i in range(M):
            add([(index_of[(u, i, j)], 1.0) for j in range(P)], 0.0, 1.0)

    # Shared stock.
    for i in range(M):
        add(
            [(index_of[(u, i, j)], 1.0) for u in range(U) for j in range(P)],
            0.0,
            float(stock[i]),
        )

    # Per-user per-category lower/upper bounds.
    for u in range(U):
        for c in range(C):
            coefficients = [
                (index_of[(u, i, j)], 1.0)
                for i in range(M)
                if category[i] == c
                for j in range(P)
            ]
            add(coefficients, float(mins[u][c]), float(maxs[u][c]))

    # An item is unavailable to a user when not allowed.
    for u in range(U):
        for i in range(M):
            if not allowed[u][i]:
                add([(index_of[(u, i, j)], 1.0) for j in range(P)], 0.0, 0.0)

    A = coo_matrix((data, (rows, cols)), shape=(row, n)).tocsr()
    constraints = LinearConstraint(A, np.asarray(lower), np.asarray(upper))
    c = -np.array([v[u, i, j] for (u, i, j) in indices], dtype=float)
    integrality = np.ones(n)
    bounds = Bounds(np.zeros(n), np.ones(n))
    result = milp(
        c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={
            "time_limit": 60.0,
            # The value tolerances are tight (rtol=1e-9), so do not accept
            # HiGHS' default relative MIP gap.
            "mip_rel_gap": 0.0,
            "presolve": True,
        },
    )
    if result.status == 2:  # infeasible
        return "infeasible", None, result
    if result.status == 3:  # unbounded (cannot happen for a 0/1 problem)
        return "error", None, result
    if result.status != 0 or result.x is None:
        return "error", None, result
    x = np.asarray(result.x, dtype=float)
    if not np.all(np.isfinite(x)):
        return "error", None, result
    x = np.clip(np.rint(x), 0.0, 1.0)
    assignment = []
    for u in range(U):
        slots = []
        for j in range(P):
            chosen = None
            for i in range(M):
                if x[index_of[(u, i, j)]] > 0.5:
                    chosen = i
                    break
            if chosen is None:
                return "error", None, result
            slots.append(chosen)
        assignment.append(slots)
    return "optimal", assignment, result


def _validate_assignment(assignment, v, allowed, stock, category, mins, maxs):
    U, M, P = v.shape
    if isinstance(assignment, np.ndarray):
        assignment = assignment.tolist()
    if not isinstance(assignment, Sequence) or len(assignment) != U:
        _fail(f"assignment must have U={U} rows")
    parsed = []
    for u, row in enumerate(assignment):
        if isinstance(row, np.ndarray):
            row = row.tolist()
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            _fail(f"assignment[{u}] must be a sequence")
        if len(row) != P:
            _fail(f"assignment[{u}] must have P={P} slots")
        slots = []
        for j, item in enumerate(row):
            integer = _as_int(item, f"assignment[{u}][{j}]")
            if integer < 0 or integer >= M:
                _fail(f"assignment[{u}][{j}]={integer} is not a valid item index")
            slots.append(integer)
        parsed.append(slots)
    if not _feasible(parsed, allowed, stock, category, mins, maxs):
        _fail("assignment violates allowed items, stock or category limits")
    return parsed


def allocate(
    values,
    allowed,
    stock,
    category,
    category_min,
    category_max,
    *,
    method="auto",
):
    """Find a maximum-value complete list allocation.

    Parameters
    ----------
    values
        ``U x M x P`` real tensor, every entry in [0, 1].  It does not have
        to come from :mod:`starter.click_model`.
    allowed
        ``U x M`` boolean table of item availability.
    stock
        Length-``M`` non-negative integers, each at most ``U``.
    category
        Length-``M`` integer item category, each in 0..2.
    category_min, category_max
        ``U x C`` integer tables with ``0 <= min <= max <= P``.

    Returns
    -------
    dict
        ``status`` is exactly ``"optimal"`` or ``"infeasible"`` on a normal
        run.  ``assignment`` is a ``U x P`` list (or ``None``), ``value`` is
        recomputed from that list, ``per_user`` is a list of lists, and
        ``objective`` is the value of the solver's list.
    """

    if method not in {"auto", "milp", "enumerate"}:
        _fail(f"unknown method {method!r}")
    v, allowed_matrix, stock_list, category_list, mins, maxs, U, M, P, C = _parse_instance(
        values, allowed, stock, category, category_min, category_max
    )
    methods_tried = []

    method_used = method
    solver_status = None
    if method == "enumerate":
        methods_tried.append("enumerate")
        best_assignment, best_value = _full_enumeration(
            v, allowed_matrix, stock_list, category_list, mins, maxs
        )
        if best_assignment is None:
            return Result(
                status="infeasible",
                assignment=None,
                value=None,
                per_user=None,
                category_counts=None,
                stock_usage=None,
                objective=None,
                method="enumerate",
                methods_tried=methods_tried,
                message="no complete legal allocation exists",
            )
        solver_status = "optimal"
    else:
        methods_tried.append("milp")
        solver_status, best_assignment, solver_result = _solve_milp(
            v, allowed_matrix, stock_list, category_list, mins, maxs
        )
        method_used = "milp"
        if solver_status == "infeasible":
            return Result(
                status="infeasible",
                assignment=None,
                value=None,
                per_user=None,
                category_counts=None,
                stock_usage=None,
                objective=None,
                method="milp",
                methods_tried=methods_tried,
                message="no complete legal allocation exists",
            )
        if solver_status != "optimal" or best_assignment is None:
            if method == "milp":
                _fail("MILP solver failed to prove an optimum or infeasibility")
            # Fall back to complete enumeration for small instances.
            methods_tried.append("enumerate")
            best_assignment, best_value = _full_enumeration(
                v, allowed_matrix, stock_list, category_list, mins, maxs
            )
            method_used = "enumerate"
            if best_assignment is None:
                return Result(
                    status="infeasible",
                    assignment=None,
                    value=None,
                    per_user=None,
                    category_counts=None,
                    stock_usage=None,
                    objective=None,
                    method=method_used,
                    methods_tried=methods_tried,
                    message="no complete legal allocation exists",
                )

    best_assignment = _validate_assignment(
        best_assignment, v, allowed_matrix, stock_list, category_list, mins, maxs
    )
    value = _objective(best_assignment, v)
    per_user = [
        [float(v[u, best_assignment[u][j], j]) for j in range(P)] for u in range(U)
    ]
    category_report = []
    stock_usage = []
    for u in range(U):
        counts = {}
        for item in best_assignment[u]:
            counts[category_list[item]] = counts.get(category_list[item], 0) + 1
        row = []
        for c in range(len(mins[u])):
            row.append(
                {
                    "category": c,
                    "count": counts.get(c, 0),
                    "min": mins[u][c],
                    "max": maxs[u][c],
                }
            )
        category_report.append(row)
    for i in range(M):
        used = sum(1 for u in range(U) for item in best_assignment[u] if item == i)
        stock_usage.append({"item": i, "used": used, "stock": stock_list[i]})
    result = Result(
        status="optimal",
        assignment=best_assignment,
        value=value,
        objective=value,
        per_user=per_user,
        category_counts=category_report,
        stock_usage=stock_usage,
        method=method_used,
        methods_tried=methods_tried,
        message="",
    )
    return result


def allocate_tensor(values, allowed, stock, category, category_min, category_max, **kwargs):
    """Alias for :func:`allocate`."""

    return allocate(
        values, allowed, stock, category, category_min, category_max, **kwargs
    )


def evaluate_assignment(
    assignment, values, allowed, stock, category, category_min, category_max
):
    """Recompute the value of an explicit assignment and check legality."""

    v, allowed_matrix, stock_list, category_list, mins, maxs, U, M, P, C = _parse_instance(
        values, allowed, stock, category, category_min, category_max
    )
    parsed = _validate_assignment(
        assignment, v, allowed_matrix, stock_list, category_list, mins, maxs
    )
    return Result(
        value=_objective(parsed, v),
        per_user=[
            [float(v[u, parsed[u][j], j]) for j in range(P)] for u in range(U)
        ],
        assignment=parsed,
    )
