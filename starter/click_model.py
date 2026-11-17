"""Position-aware click model with a monotone examination cascade.

The model is

    r(x)   = sigmoid(x . w)
    e[0]   = 1
    e[j]   = prod_{k=1..j} sigmoid(s[k-1])
    p(x,j) = e[j] * r(x)

Only ``s[0..P-2]`` is stored; position zero fixes the examination scale at
one.  A row of a display log is a mapping with keys ``features`` (1..4 finite
numbers in [-1, 1]), ``click`` (0 or 1), ``position`` (0-based) and
``weight`` (positive integer).  Repeated rows are legitimate observations and
their weights are simply accumulated by the likelihood.

The penalised negative log likelihood is

    sum_i weight_i * [-click_i log p_i - (1-click_i) log(1-p_i)]
    + ridge/2 * (||w||^2 + ||s||^2)

with ``w`` and ``s`` restricted to the box [-3, 3].  Integers are never
rounded and no illegal value is silently clipped: all validation failures
raise ``ValueError``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

PARAM_MIN = -3.0
PARAM_MAX = 3.0
RIDGE_MIN = 0.01
RIDGE_MAX = 1.0
PROJECTED_GRADIENT_TOL = 1e-5
MAX_ROWS = 200
MAX_FEATURES = 4
MAX_POSITIONS = 3
MAX_WEIGHT = 50


class Result(dict):
    """A dict whose entries are also readable as attributes."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - normal attribute miss
            raise AttributeError(name) from exc


def _fail(message: str) -> None:
    raise ValueError(message)


def _is_real_number(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )


def _is_integer_value(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (int, np.integer)):
        return True
    return isinstance(value, (float, np.floating)) and math.isfinite(float(value)) and float(
        value
    ).is_integer()


def _as_int(value, name: str) -> int:
    if not _is_integer_value(value):
        _fail(f"{name} must be an integer, got {value!r}")
    return int(value)


def _validate_ridge(ridge) -> float:
    if not _is_real_number(ridge):
        _fail(f"ridge must be a real number, got {ridge!r}")
    value = float(ridge)
    if not math.isfinite(value) or value < RIDGE_MIN or value > RIDGE_MAX:
        _fail(f"ridge must be finite and in [{RIDGE_MIN}, {RIDGE_MAX}], got {ridge!r}")
    return value


def _validate_click(value) -> float:
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if not _is_real_number(value):
        _fail(f"click must be 0 or 1, got {value!r}")
    value = float(value)
    if value not in (0.0, 1.0):
        _fail(f"click must be 0 or 1, got {value!r}")
    return value


def _validate_weight(value) -> float:
    if not _is_real_number(value):
        _fail(f"weight must be a positive integer, got {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        _fail(f"weight must be a positive integer, got {value!r}")
    integer = int(numeric)
    if integer < 1 or integer > MAX_WEIGHT:
        _fail(f"weight must be in 1..{MAX_WEIGHT}, got {value!r}")
    return float(integer)


def _validate_features(value, row_index: int) -> np.ndarray:
    try:
        features = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_index}: features must be a numeric vector") from exc
    if features.ndim != 1:
        _fail(f"row {row_index}: features must be one-dimensional")
    if features.size < 1 or features.size > MAX_FEATURES:
        _fail(
            f"row {row_index}: features dimension must be in 1..{MAX_FEATURES}, "
            f"got {features.size}"
        )
    if not np.all(np.isfinite(features)):
        _fail(f"row {row_index}: features contain a non-finite value")
    if np.any(features < -1.0) or np.any(features > 1.0):
        _fail(f"row {row_index}: every feature component must be in [-1, 1]")
    return features


def _parse_rows(rows, dimension=None, positions=None):
    """Validate display rows and return (X, click, position, weight, D, P)."""

    if isinstance(rows, Mapping):
        # Accept a columnar mapping: features, click, position and weight are
        # parallel sequences.
        for key in ("features", "click", "position", "weight"):
            if key not in rows:
                _fail(f"rows mapping is missing {key!r}")
        lengths = {
            len(rows["features"]),
            len(rows["click"]),
            len(rows["position"]),
            len(rows["weight"]),
        }
        if len(lengths) != 1:
            _fail("columnar rows have inconsistent lengths")
        rows = [
            {
                "features": rows["features"][index],
                "click": rows["click"][index],
                "position": rows["position"][index],
                "weight": rows["weight"][index],
            }
            for index in range(lengths.pop())
        ]
    elif isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        try:
            rows = list(rows)
        except TypeError as exc:
            raise ValueError("rows must be a sequence of mappings") from exc
    else:
        rows = list(rows)
    if len(rows) < 1 or len(rows) > MAX_ROWS:
        _fail(f"rows length must be in 1..{MAX_ROWS}, got {len(rows)}")

    features = []
    clicks = []
    position_values = []
    weights = []

    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            _fail(f"row {index} must be a mapping")
        for key in ("features", "click", "position", "weight"):
            if key not in row:
                _fail(f"row {index} is missing {key!r}")
        features.append(_validate_features(row["features"], index))
        clicks.append(_validate_click(row["click"]))
        position_values.append(_as_int(row["position"], f"row {index}: position"))
        weights.append(_validate_weight(row["weight"]))

    inferred_dimension = features[0].size
    if dimension is None:
        dimension = inferred_dimension
    else:
        dimension = _as_int(dimension, "dimension")
        if dimension < 1 or dimension > MAX_FEATURES:
            _fail(f"dimension must be in 1..{MAX_FEATURES}, got {dimension}")
    for index, row_features in enumerate(features):
        if row_features.size != dimension:
            _fail(
                f"row {index}: feature dimension {row_features.size} does not match "
                f"the requested dimension {dimension}"
            )

    if positions is None:
        positions = max(position_values) + 1
    else:
        positions = _as_int(positions, "positions")
    if positions < 1 or positions > MAX_POSITIONS:
        _fail(f"positions must be in 1..{MAX_POSITIONS}, got {positions}")
    for index, position in enumerate(position_values):
        if position < 0 or position >= positions:
            _fail(
                f"row {index}: position {position} is not in 0..{positions - 1}"
            )

    X = np.vstack(features)
    y = np.asarray(clicks, dtype=float)
    pos = np.asarray(position_values, dtype=int)
    weight = np.asarray(weights, dtype=float)
    return X, y, pos, weight, int(dimension), int(positions)


def _unpack_theta(theta, dimension: int, positions: int) -> np.ndarray:
    try:
        theta = np.asarray(theta, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("theta must be a numeric vector") from exc
    if theta.ndim != 1:
        _fail("theta must be one-dimensional")
    expected = dimension + (positions - 1)
    if theta.size != expected:
        _fail(
            f"theta must have length {expected} (w: {dimension}, s: "
            f"{positions - 1}), got {theta.size}"
        )
    if not np.all(np.isfinite(theta)):
        _fail("theta contains a non-finite value")
    if np.any(theta < PARAM_MIN) or np.any(theta > PARAM_MAX):
        _fail(f"every parameter must be in [{PARAM_MIN}, {PARAM_MAX}]")
    return theta.copy()


def _core_objective(theta, X, y, pos, weight, ridge, dimension, positions):
    """Objective and exact gradient for already validated arrays.

    Gradient ordering is ``[w..., s...]``.  The loss is evaluated in a form
    that avoids cancellation when position zero has a large sigmoid.
    """

    w = theta[:dimension]
    s = theta[dimension:]
    n = X.shape[0]

    z = X @ w
    r = expit(z)
    log_r = -np.logaddexp(0.0, -z)

    if positions == 1:
        e = np.ones(n, dtype=float)
        log_e = np.zeros(n, dtype=float)
        t = np.empty(0, dtype=float)
    else:
        t = expit(s)
        log_t = -np.logaddexp(0.0, -s)
        e = np.ones(n, dtype=float)
        log_e = np.zeros(n, dtype=float)
        for k in range(1, positions):
            mask = pos >= k
            if np.any(mask):
                e[mask] *= t[k - 1]
                log_e[mask] += log_t[k - 1]

    p = e * r
    log_p = log_r + log_e

    # log(1-p), written to avoid cancellation at position zero.
    log_one_minus_p = np.empty(n, dtype=float)
    zero_position = pos == 0
    if np.any(zero_position):
        log_one_minus_p[zero_position] = -np.logaddexp(0.0, z[zero_position])
    if np.any(~zero_position):
        # At positions >= 1, e <= sigmoid(3) < 1, so 1-p is not close to zero.
        log_one_minus_p[~zero_position] = np.log1p(-p[~zero_position])

    row_loss = -(y * log_p + (1.0 - y) * log_one_minus_p)
    objective = float(np.dot(weight, row_loss))
    objective += float(ridge) / 2.0 * (np.dot(w, w) + np.dot(s, s))

    # d loss / dp times p/(1-p) is weight*(p-y)/(1-p).  Multiplying by the
    # derivative of p with respect to z or s gives the compact expressions
    # below.  Inside the box, p <= sigmoid(3) < 1, so the denominator is
    # bounded away from zero and this form is numerically safe.
    ratio = weight * (p - y) / (1.0 - p)
    coeff_z = ratio * (1.0 - r)
    grad_w = X.T @ coeff_z + ridge * w

    grad_s = np.zeros(max(positions - 1, 0), dtype=float)
    for k in range(1, positions):
        mask = pos >= k
        if np.any(mask):
            grad_s[k - 1] = np.sum(ratio[mask] * (1.0 - t[k - 1]))
    if positions > 1:
        grad_s += ridge * s

    gradient = np.concatenate([grad_w, grad_s])
    return objective, gradient


def _looks_like_rows(value) -> bool:
    if isinstance(value, Mapping):
        return set(("features", "click", "position", "weight")).issubset(value)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return False
    if len(value) == 0:
        # An empty list is accepted as the (invalid) row collection so that
        # the normal validator can report a clear length error.
        return True
    return isinstance(value[0], Mapping)


def _parse_objective_call(args, kwargs):
    """Accept both ``objective(theta, rows, ...)`` and ``objective(w, s, rows, ...)``.

    The split form is useful for hand-written checks.  A list of mappings is
    unambiguously display rows, whereas a numeric vector is unambiguously a
    parameter vector.
    """

    kwargs = dict(kwargs)
    if "theta" in kwargs:
        theta = kwargs.pop("theta")
        if not args:
            rows = kwargs.pop("rows", _MISSING)
        else:
            rows = args[0]
            args = args[1:]
    else:
        if len(args) < 2:
            _fail("objective needs a parameter vector and rows")
        theta = args[0]
        if _looks_like_rows(args[1]):
            rows = args[1]
            args = args[2:]
        elif len(args) >= 3 and _looks_like_rows(args[2]):
            w = np.asarray(args[0], dtype=float).ravel()
            s = np.asarray(args[1], dtype=float).ravel()
            theta = np.concatenate([w, s])
            rows = args[2]
            args = args[3:]
        else:
            _fail("objective could not identify the display rows")
    if rows is _MISSING:
        _fail("objective needs display rows")

    names = ("ridge", "positions", "dimension")
    if len(args) > len(names):
        _fail(f"objective received too many positional arguments ({len(args)} extra)")
    for name, value in zip(names, args):
        if name in kwargs:
            _fail(f"objective received {name!r} twice")
        kwargs[name] = value

    ridge = kwargs.pop("ridge", 0.1)
    positions = kwargs.pop("positions", kwargs.pop("P", None))
    dimension = kwargs.pop("dimension", kwargs.pop("D", None))
    if kwargs:
        _fail(f"unrecognised objective arguments: {sorted(kwargs)}")
    return theta, rows, ridge, positions, dimension


_MISSING = object()


def objective(*args, **kwargs):
    """Return ``(objective, gradient)`` for the position-aware model.

    The usual call is ``objective(theta, rows, ridge=0.1)`` where ``theta`` is
    the concatenation ``[w, s]``.  ``objective(w, s, rows, ...)`` is also
    accepted.  When ``positions`` and ``dimension`` are omitted they are
    inferred from the rows.  The returned gradient is ordered in exactly the
    same way as ``theta`` (w first, then s).
    """

    theta, rows, ridge, positions, dimension = _parse_objective_call(args, kwargs)
    ridge = _validate_ridge(ridge)
    X, y, pos, weight, dimension, positions = _parse_rows(
        rows, dimension=dimension, positions=positions
    )
    theta = _unpack_theta(theta, dimension, positions)
    return _core_objective(theta, X, y, pos, weight, ridge, dimension, positions)


def gradient(theta, rows, ridge=0.1, positions=None, dimension=None):
    """Return only the exact gradient, ordered ``[w, s]``."""

    return objective(theta, rows, ridge, positions=positions, dimension=dimension)[1]


def full_gradient(theta, rows, ridge=0.1, positions=None, dimension=None):
    """Alias for :func:`gradient`."""

    return gradient(theta, rows, ridge, positions=positions, dimension=dimension)


def objective_split(w, s, rows, ridge=0.1, positions=None, dimension=None):
    """Objective/gradient with separate ``w`` and ``s`` arguments."""

    w = np.asarray(w, dtype=float).ravel()
    s = np.asarray(s, dtype=float).ravel()
    return objective(np.concatenate([w, s]), rows, ridge, positions, dimension)


def examination(s, position):
    """Return examination probability at a 0-based position.

    ``s`` may be the examination-slope vector (length ``positions - 1``) or a
    full mapping with an ``s`` key.  ``position`` may be an integer (returning
    a scalar) or an array of positions (returning an array).
    """

    if isinstance(s, Mapping):
        s = s.get("s", [])
    s = np.asarray(s, dtype=float).ravel()
    if not np.all(np.isfinite(s)):
        _fail("s contains a non-finite value")
    if np.any(s < PARAM_MIN) or np.any(s > PARAM_MAX):
        _fail(f"every s component must be in [{PARAM_MIN}, {PARAM_MAX}]")
    positions = s.size + 1

    slopes = np.array([expit(value) for value in s], dtype=float)

    def one(value):
        if not _is_integer_value(value):
            _fail(f"position must be an integer, got {value!r}")
        value = int(value)
        if value < 0 or value >= positions:
            _fail(f"position {value} is not in 0..{positions - 1}")
        result = 1.0
        for k in range(1, value + 1):
            result *= float(slopes[k - 1])
        return float(result)

    array = np.asarray(position)
    if array.ndim == 0:
        return one(array.item())
    if not np.all(np.isfinite(array.astype(float))) or not np.all(
        array == np.rint(array)
    ):
        _fail("positions must be finite integers")
    return np.array([one(int(value)) for value in array.ravel()], dtype=float).reshape(
        array.shape
    )


def _query_features(features, dimension: int) -> np.ndarray:
    """Validate a feature vector (or array of vectors) for prediction.

    A scalar or a flat vector is treated as a single feature vector.  The
    components must be finite and in [-1, 1], matching the learning domain.
    """

    try:
        x = np.asarray(features, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("features must be numeric") from exc
    if x.ndim == 0:
        x = x.reshape(1)
    if x.ndim == 1:
        if x.size != dimension:
            _fail(
                f"feature dimension {x.size} does not match w dimension {dimension}"
            )
    elif x.shape[-1] != dimension:
        _fail(
            f"feature dimension {x.shape[-1]} does not match w dimension {dimension}"
        )
    if not np.all(np.isfinite(x)):
        _fail("features contain a non-finite value")
    if np.any(x < -1.0) or np.any(x > 1.0):
        _fail("every feature component must be in [-1, 1]")
    return x


def relevance(w, features) -> np.ndarray:
    """Return ``sigmoid(x . w)`` for one or more feature vectors."""

    w = np.asarray(w, dtype=float).ravel()
    if w.size < 1:
        _fail("w must be non-empty")
    if not np.all(np.isfinite(w)):
        _fail("w contains a non-finite value")
    if np.any(w < PARAM_MIN) or np.any(w > PARAM_MAX):
        _fail(f"every w component must be in [{PARAM_MIN}, {PARAM_MAX}]")
    x = _query_features(features, w.size)
    return expit(x @ w)


def _model_parts(model):
    if not isinstance(model, Mapping):
        _fail("model must be a mapping with w and s")
    if "w" not in model:
        _fail("model is missing 'w'")
    w = np.asarray(model["w"], dtype=float).ravel()
    s = np.asarray(model.get("s", []), dtype=float).ravel()
    if w.size < 1:
        _fail("model w must be non-empty")
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(s)):
        _fail("model parameters must be finite")
    if np.any(w < PARAM_MIN) or np.any(w > PARAM_MAX):
        _fail(f"every w component must be in [{PARAM_MIN}, {PARAM_MAX}]")
    if np.any(s < PARAM_MIN) or np.any(s > PARAM_MAX):
        _fail(f"every s component must be in [{PARAM_MIN}, {PARAM_MAX}]")
    if s.size > MAX_POSITIONS - 1:
        _fail(f"s has length {s.size}, but at most {MAX_POSITIONS - 1} is allowed")
    return w, s


def probabilities(model, features, position=None):
    """Return relevance, examination and click probabilities.

    The result always contains all three keys ``r``, ``e`` and ``p``.  ``p`` is
    the product ``e * r``; it is never reported as relevance.  If ``position``
    is given, scalars (for a single feature vector) or arrays are returned.  If
    it is omitted, ``e`` and ``p`` have a trailing position axis.
    """

    w, s = _model_parts(model)
    x = _query_features(features, w.size)
    positions = s.size + 1
    r = expit(x @ w)

    if position is not None:
        if not _is_integer_value(position):
            _fail(f"position must be an integer, got {position!r}")
        position = int(position)
        if position < 0 or position >= positions:
            _fail(f"position {position} is not in 0..{positions - 1}")
        e = examination(s, position)
        p = e * r
        if np.ndim(r) == 0:
            return {"r": float(r), "e": float(e), "p": float(p)}
        return {"r": r, "e": np.full_like(r, float(e)), "p": p}

    e = np.array([examination(s, j) for j in range(positions)], dtype=float)
    if np.ndim(r) == 0:
        return {"r": float(r), "e": e, "p": float(r) * e}
    # Broadcast trailing position axis.
    return {"r": r, "e": e, "p": np.expand_dims(r, axis=-1) * e}


def predict_proba(model, features, position=None):
    """Alias for :func:`probabilities`."""

    return probabilities(model, features, position)


def predict_probabilities(model, features, position=None):
    """Alias for :func:`probabilities`."""

    return probabilities(model, features, position)


def click_probability(model, features, position=None):
    """Return ``(r, e, p)`` in that order for one query."""

    result = probabilities(model, features, position)
    return result["r"], result["e"], result["p"]


def value_tensor(model, features):
    """Return ``v[u, i, j] = e[j] * sigmoid(x[u, i] . w)``.

    ``features`` must be a ``U x M x D`` array (or a list of lists of
    vectors).  The third axis may be omitted for a single position, in which
    case a ``U x M x 1`` tensor is returned.
    """

    w, s = _model_parts(model)
    positions = s.size + 1
    x = np.asarray(features, dtype=float)
    if x.ndim == 2:
        x = x[:, :, None] if w.size == 1 else x
    if x.ndim != 3:
        _fail("features must be a U x M x D array")
    if x.shape[-1] != w.size:
        _fail(
            f"feature dimension {x.shape[-1]} does not match w dimension {w.size}"
        )
    if not np.all(np.isfinite(x)):
        _fail("features contain a non-finite value")
    if np.any(x < -1.0) or np.any(x > 1.0):
        _fail("every feature component must be in [-1, 1]")
    r = expit(x @ w)
    e = np.array([examination(s, j) for j in range(positions)], dtype=float)
    return r[:, :, None] * e[None, None, :]


def predict(model, features, position=None):
    """Return the click probability only (alias retained for convenience)."""

    return probabilities(model, features, position)["p"]


def _normalise_initial(initial, dimension, positions):
    length = dimension + positions - 1
    if initial is None:
        return np.zeros(length, dtype=float)

    if isinstance(initial, Mapping):
        if "w" not in initial:
            _fail("initial must contain 'w'")
        w = np.asarray(initial["w"], dtype=float).ravel()
        s = np.asarray(initial.get("s", []), dtype=float).ravel()
        if w.size != dimension:
            _fail(
                f"initial w must have length {dimension}, got {w.size}"
            )
        if s.size != positions - 1:
            if s.size == 0:
                # A w-only initial is a reasonable shorthand; the missing
                # examination slopes start at zero.
                s = np.zeros(positions - 1, dtype=float)
            else:
                _fail(
                    f"initial s must have length {positions - 1}, got {s.size}"
                )
        theta = np.concatenate([w, s])
        return _unpack_theta(theta, dimension, positions)

    if isinstance(initial, np.ndarray):
        values = np.asarray(initial, dtype=float)
    else:
        try:
            items = list(initial)
        except TypeError as exc:
            raise ValueError("initial must be a mapping or a numeric vector") from exc
        if len(items) == 2 and all(
            isinstance(item, (Sequence, np.ndarray)) and not isinstance(item, (str, bytes))
            for item in items
        ):
            w = np.asarray(items[0], dtype=float).ravel()
            s = np.asarray(items[1], dtype=float).ravel()
            theta = np.concatenate([w, s])
            return _unpack_theta(theta, dimension, positions)
        values = np.asarray(items, dtype=float)

    return _unpack_theta(values.ravel(), dimension, positions)


def _projected_gradient_residual(theta, grad):
    if theta.size == 0:
        return 0.0
    projected = np.clip(theta - grad, PARAM_MIN, PARAM_MAX)
    return float(np.max(np.abs(theta - projected)))


def _status_from_optimisation(residual, optimizer_message, optimizer_status, finite):
    if not finite:
        return "numerical_failure", False
    if residual <= PROJECTED_GRADIENT_TOL:
        return "converged", True
    text = str(optimizer_message).lower()
    if "nan" in text or "inf" in text or "abnormal" in text or "error" in text:
        return "numerical_failure", False
    if "iteration" in text or "limit" in text or optimizer_status == 1:
        return "max_iter", False
    return "not_converged", False


def fit(rows, dimension=None, ridge=0.1, positions=None, initial=None, **options):
    """Fit the position-aware model inside the box ``[-3, 3]``.

    ``initial`` may be ``{"w": [...], "s": [...]}`` or a flat ``[w, s]``
    vector.  The function returns a mapping with the current model, its true
    objective, the exact gradient, the projected-gradient residual and an
    explicit stop status.  ``success`` from the optimizer is never used as the
    convergence criterion: only a projected-gradient residual at most ``1e-5``
    is reported as ``converged``.
    """

    # Be liberal about positional call forms while keeping keyword use
    # unambiguous:
    #   fit(rows, D, ridge)
    #   fit(rows, D, initial)
    #   fit(rows, D, P, ridge, initial)
    #   fit(rows, initial)
    if initial is None and isinstance(ridge, Mapping):
        initial, ridge = ridge, 0.1
    elif initial is None and isinstance(ridge, np.ndarray):
        initial, ridge = ridge, 0.1
    elif initial is None and isinstance(ridge, (Sequence, np.ndarray)) and not isinstance(
        ridge, (str, bytes)
    ):
        initial, ridge = ridge, 0.1

    # fit(rows, D, ridge, initial): initial landed in the positions slot.
    if (
        initial is None
        and positions is not None
        and isinstance(positions, (Mapping, np.ndarray, Sequence))
        and not isinstance(positions, (str, bytes))
        and not _is_integer_value(positions)
    ):
        initial, positions = positions, None

    if isinstance(dimension, Mapping) or (
        isinstance(dimension, (Sequence, np.ndarray))
        and not isinstance(dimension, (str, bytes))
        and not _is_integer_value(dimension)
    ):
        if initial is not None:
            _fail("initial was supplied more than once")
        initial = dimension
        dimension = None

    if positions is None and _is_integer_value(ridge) and int(ridge) in (2, 3):
        # An integer 2 or 3 in the ridge slot can never be a legal ridge (the
        # legal range is [0.01, 1]), so it is unambiguously the position count.
        # An integer 1 is left alone because 1 is a legal ridge value; pass
        # positions=1 explicitly for a P=1 model.
        positions, ridge = int(ridge), 0.1
    if positions is not None and _is_real_number(positions) and not _is_integer_value(
        positions
    ):
        # Call form fit(rows, D, P, ridge, initial): P got bound to ridge and
        # the ridge value got bound to positions.
        if _is_integer_value(ridge):
            positions, ridge = int(ridge), float(positions)
        else:
            _fail(f"positions must be an integer, got {positions!r}")

    ridge = _validate_ridge(ridge)
    X, y, pos, weight, dimension, positions = _parse_rows(
        rows, dimension=dimension, positions=positions
    )
    theta0 = _normalise_initial(initial, dimension, positions)

    max_iter = int(options.pop("max_iter", options.pop("maxiter", 10000)))
    if max_iter < 1:
        _fail("max_iter must be positive")
    gtol = float(options.pop("gtol", 1e-12))
    ftol = float(options.pop("ftol", 0.0))
    maxls = int(options.pop("maxls", 50))
    if options:
        _fail(f"unrecognised fit options: {sorted(options)}")
    if not all(math.isfinite(v) for v in (gtol, ftol)):
        _fail("gtol and ftol must be finite")
    if gtol < 0.0 or ftol < 0.0:
        _fail("gtol and ftol must be non-negative")

    bounds = [(PARAM_MIN, PARAM_MAX)] * theta0.size

    def fun(theta):
        return _core_objective(theta, X, y, pos, weight, ridge, dimension, positions)

    initial_value = fun(theta0)[0]
    if not math.isfinite(initial_value):
        _fail("the objective is not finite at the initial parameters")

    best_theta = theta0.copy()
    best_value = initial_value
    best_gradient = fun(best_theta)[1]
    finite = True
    message = ""
    status = 0
    nit = 0

    try:
        solution = minimize(
            fun,
            theta0,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": max_iter,
                "maxls": maxls,
                "ftol": ftol,
                "gtol": gtol,
                "maxcor": 30,
                "maxfun": max(15000, 100 * theta0.size),
            },
        )
        candidate = np.asarray(solution.x, dtype=float)
        message = str(solution.message)
        status = int(solution.status)
        nit = int(getattr(solution, "nit", 0))
        if (
            candidate.ndim == 1
            and candidate.size == theta0.size
            and np.all(np.isfinite(candidate))
            and np.all(candidate >= PARAM_MIN)
            and np.all(candidate <= PARAM_MAX)
        ):
            candidate_value, candidate_gradient = fun(candidate)
            if math.isfinite(candidate_value) and candidate_value <= best_value:
                best_theta = candidate.copy()
                best_value = float(candidate_value)
                best_gradient = candidate_gradient
        elif not np.all(np.isfinite(candidate)):
            finite = False
    except (FloatingPointError, OverflowError, ValueError) as exc:
        finite = False
        message = f"numerical failure: {exc}"

    # A second, independent local pass from the best point is cheap here and
    # helps L-BFGS-B reach a tight projected-gradient residual on flat ridges.
    if finite:
        residual = _projected_gradient_residual(best_theta, best_gradient)
        if residual > PROJECTED_GRADIENT_TOL:
            try:
                solution2 = minimize(
                    fun,
                    best_theta,
                    jac=True,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={
                        "maxiter": max_iter,
                        "maxls": maxls,
                        "ftol": ftol,
                        "gtol": gtol,
                        "maxcor": 50,
                        "maxfun": max(15000, 100 * theta0.size),
                    },
                )
                candidate = np.asarray(solution2.x, dtype=float)
                if (
                    candidate.ndim == 1
                    and candidate.size == theta0.size
                    and np.all(np.isfinite(candidate))
                    and np.all(candidate >= PARAM_MIN)
                    and np.all(candidate <= PARAM_MAX)
                ):
                    candidate_value, candidate_gradient = fun(candidate)
                    if math.isfinite(candidate_value) and candidate_value <= best_value:
                        best_theta = candidate.copy()
                        best_value = float(candidate_value)
                        best_gradient = candidate_gradient
                        message = str(solution2.message)
                        status = int(solution2.status)
                        nit += int(getattr(solution2, "nit", 0))
            except (FloatingPointError, OverflowError, ValueError) as exc:
                finite = False
                message = f"numerical failure: {exc}"

    finite = finite and np.all(np.isfinite(best_theta)) and math.isfinite(best_value)
    residual = (
        _projected_gradient_residual(best_theta, best_gradient) if finite else float("inf")
    )
    stop_status, converged = _status_from_optimisation(residual, message, status, finite)

    w = best_theta[:dimension]
    s = best_theta[dimension:]
    result = Result(
        w=w.tolist(),
        s=s.tolist(),
        theta=best_theta.tolist(),
        objective=float(best_value),
        gradient=best_gradient.tolist(),
        projected_gradient_residual=float(residual),
        gradient_residual=float(residual),
        pg_residual=float(residual),
        converged=bool(converged),
        success=bool(converged),
        status=stop_status,
        stop_status=stop_status,
        iterations=int(nit),
        nit=int(nit),
        message=message,
        initial_objective=float(initial_value),
        initial=(
            {"w": theta0[:dimension].tolist(), "s": theta0[dimension:].tolist()}
        ),
        dimension=int(dimension),
        positions=int(positions),
    )
    return result


def fit_position_aware(rows, dimension=None, ridge=0.1, positions=None, initial=None, **options):
    """Alias for :func:`fit`."""

    return fit(
        rows,
        dimension=dimension,
        ridge=ridge,
        positions=positions,
        initial=initial,
        **options,
    )
