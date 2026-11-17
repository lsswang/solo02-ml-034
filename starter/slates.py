"""Per-user and joint recommendation lists.

This module keeps the original independent recommender and adds a joint
helper above :func:`starter.allocation.allocate`.  In all cases the value
tensor is ``U x M x P``: the value of placing item ``i`` in slot ``j`` for
user ``u``.  Values are click probabilities; they are never rescaled or
discretised here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np

from .allocation import allocate


class InputError(ValueError):
    """Raised for illegal value-tensor shapes or indices."""


def _as_values(values):
    try:
        v = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InputError("values must be numeric") from exc
    if not np.all(np.isfinite(v)):
        raise InputError("values contains a non-finite entry")
    return v


def recommend(values, slots):
    """Independent per-user greedy lists, sorted by value at each slot.

    The existing relevance-only behaviour is preserved for a ``U x M`` matrix:
    ``slots`` items are selected per user and ordered by value.  For a
    ``U x M x P`` tensor the value of item ``i`` in slot ``j`` is used
    directly; this ignores shared stock and is provided only as the baseline
    that the joint allocator improves on.
    """

    if isinstance(slots, bool) or not isinstance(slots, (int, np.integer)):
        raise InputError("slots must be an integer")
    slots = int(slots)
    if slots < 1:
        raise InputError("slots must be positive")

    v = _as_values(values)
    if v.ndim == 2:
        U, M = v.shape
        if M < slots:
            raise InputError(
                f"cannot fill {slots} slots from {M} items without repeats"
            )
        if np.any(v < 0.0) or np.any(v > 1.0):
            raise InputError("every value must be in [0, 1]")
        lists = []
        for u in range(U):
            order = sorted(range(M), key=lambda i: (-v[u, i], i))[:slots]
            lists.append(order)
        return lists

    if v.ndim != 3:
        raise InputError("values must be a U x M (or U x M x P) array")
    U, M, P = v.shape
    if np.any(v < 0.0) or np.any(v > 1.0):
        raise InputError("every value must be in [0, 1]")
    if slots > P:
        raise InputError(f"slots={slots} exceeds the value-tensor positions P={P}")
    if M < P:
        raise InputError(
            f"P={P} distinct items per user are required but only M={M} exist"
        )
    lists = []
    for u in range(U):
        # Greedy: independently take the best item for each slot.
        chosen = []
        for j in range(P):
            order = sorted(range(M), key=lambda i: (-v[u, i, j], i))
            for i in order:
                if i not in chosen:
                    chosen.append(i)
                    break
        lists.append(chosen)
    return lists


def recommend_slates(
    values,
    *,
    allowed=None,
    stock=None,
    category=None,
    category_min=None,
    category_max=None,
    method="auto",
):
    """Joint recommendation entry point.

    Parameters mirror :func:`starter.allocation.allocate`.  When shared
    constraints are omitted, every item is allowed at every user, stock is
    unlimited (``U``), all items share category zero, and the category bounds
    are ``0..P``.  The returned mapping contains the chosen lists, the
    quantity and a ``status`` of ``"optimal"`` or ``"infeasible"``.
    """

    v = _as_values(values)
    if v.ndim != 3:
        raise InputError("values must be a U x M x P array for joint allocation")
    U, M, P = v.shape
    if allowed is None:
        allowed = np.ones((U, M), dtype=bool)
    if stock is None:
        stock = [U] * M
    if category is None:
        category = [0] * M
    category_values = np.asarray(category).ravel().tolist() if not isinstance(
        category, (str, bytes)
    ) else category
    C = max(category_values) + 1 if category_values else 1
    if category_min is None:
        category_min = [[0] * C for _ in range(U)]
    if category_max is None:
        category_max = [[P] * C for _ in range(U)]

    result = allocate(
        v,
        allowed,
        stock,
        category,
        category_min,
        category_max,
        method=method,
    )
    lists = result.get("assignment")
    return {
        "status": result["status"],
        "lists": lists,
        "assignment": lists,
        "value": result.get("value"),
        "per_user": result.get("per_user"),
        "category_counts": result.get("category_counts"),
        "stock_usage": result.get("stock_usage"),
        "method": result.get("method"),
        "message": result.get("message", ""),
    }
