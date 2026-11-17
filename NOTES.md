# Method notes and measured results

## 1. Position-aware click model

Model (`starter/click_model.py`):

```
r(x) = sigmoid(x · w)
e[0] = 1,   e[j] = Π_{k=1..j} sigmoid(s[k−1])          (s has P−1 entries)
p(x,j) = e[j] · r(x)
```

Position 0 is fixed to examination probability 1. This fixes the overall
exposure scale: adding a constant to all examination logits would otherwise
trade off against `w`, so there is no separately fitted exposure coefficient.
For `P=1` the parameter vector `s` is empty.

Objective (weighted negative log likelihood plus ridge):

```
Σ_i weight_i · [−click_i log p_i − (1−click_i) log(1−p_i)]
+ ridge/2 · (||w||² + ||s||²),    ridge ∈ [0.01, 1]
```

The loss is evaluated as `log p = log r + log e` and, at position 0,
`log(1−p) = −logaddexp(0, z)`, avoiding cancellation when `sigmoid(x·w)` is
near one. The analytic gradient, ordered `[w, s]`, is

```
∂/∂w = Xᵀ[ weight·(p−y)/(1−p)·(1−r) ] + ridge·w
∂/∂s_k = Σ_{rows at position ≥ k} weight·(p−y)/(1−p)·(1−t_k) + ridge·s_k
```

with `t_k = sigmoid(s_k)`. It is checked against central finite differences
at `atol=2e-6 / rtol=2e-5` in `verify.py`.

### Optimisation and stopping

L-BFGS-B with explicit bounds `[-3, 3]` on every `w` and `s` entry,
`gtol=1e-12`, `ftol=0`, `maxiter` default 10000. The legal initial vector is
used as-is (default zeros). The result keeps the best *legal* point seen. A
second pass starts from the best point when the first pass did not reach the
target residual.

Stopping is judged only by the **projected-gradient residual**

```
max |θ − clip(θ − g, −3, 3)|
```

and reported `converged` only when it is `<= 1e-5`. The optimizer's own
`success` flag is ignored. Otherwise the status is `max_iter`,
`not_converged` or `numerical_failure`, and `converged`/`success` are false
while the best legal model is still returned.

`fit` returns `w`, `s`, `theta`, the true `objective` evaluated at the
returned model, `gradient`, `projected_gradient_residual`, `initial_objective`
and an explicit `status`. `probabilities` returns `r`, `e` and `p`
independently; `p` is always `e·r`, never reported as `r`.

Default settings converge on the supplied `fixtures/train.json` and on 120
random synthetic instances (D=1..4, P=1..3, up to 200 rows, random legal
initial points, ridge in [0.01, 1]) with worst projected residual below
`1e-5`. A one-iteration budget is reported as `max_iter`, not converged.

## 2. Joint allocation

`starter/allocation.py::allocate(values, allowed, stock, category,
category_min, category_max)` accepts any legal `U×M×P` tensor with entries in
`[0,1]`; no log or fitted model is needed. The formulation is a single MILP
(binary `x[u,i,j]`):

* exactly one item per user per position: `Σ_i x[u,i,j] = 1`;
* no repeated item per user: `Σ_j x[u,i,j] <= 1`;
* shared stock: `Σ_{u,j} x[u,i,j] <= stock[i]`;
* per-user category bounds: `min[u,c] <= Σ_{i∈c,j} x[u,i,j] <= max[u,c]`;
* disallowed pairs fixed to zero.

Maximise `Σ v[u,i,j]·x[u,i,j]`. Solver: SciPy `milp` (HiGHS) with
`mip_rel_gap=0` so the answer is proved optimal at the required
`atol=1e-8 / rtol=1e-9`. `status` is `optimal` or `infeasible`; `P>M` is
infeasible because an item cannot be repeated within a user's list. The
reported `value` is always recomputed by walking the actual assignment, and
`evaluate_assignment` re-checks legality independently.

An independent exact solver (`method="enumerate"`) enumerates every legal
per-user permutation and then every cross-user combination with stock
pruning. `verify.py` compares MILP and brute-force status and value on 120
random instances (no mismatches) plus a hand-built infeasible case and a
naive fixed-order greedy failure.

Complexity: MILP is exact and fast at the permitted sizes (measured worst
case ~0.02 s for U=6, M=8, P=3). The value tensor drives a pure assignment
problem; there is no Benders/column-generation loop. The independent
enumeration is exponential (roughly `(M!/(M−P)!)ᵁ` before pruning) and is
meant for small cross-checks.

## 3. Synthetic experiment (`python3 -m starter.experiments`)

Truth for the first evidence set: `w = [1.1, −0.8]`,
`s = [−1.2, −0.9]` (three positions). 200 weighted rows with clicks sampled
from `p = e[j]·r(x)`. Results:

* position-aware fit: `w ≈ [1.137, −0.963]`, `s ≈ [−0.476, −0.902]`,
  residual `1.5e-8`, converged, objective `1129.3` from initial `1256.7`;
* relevance-only fit (ignores position, treats every unclick as dislike):
  `w ≈ [0.60, −0.24]` – badly attenuated because deep-position impressions
  are mostly unexamined. Raw click rates by position in this sample are
  `0.524 / 0.154 / 0.042`, so a position-blind model partly learns the
  examination cascade as if it were a per-feature dislike;
* examination probabilities recovered as `e = [1.0, 0.383, 0.111]`.

Six users, eight items, three slots, stock `[2,1,3,3,3,2,2,2]`, category
mixes with a per-user category-2 minimum. The joint optimum value is
`6.1608`; the fixed-user-order greedy cannot even produce a complete legal
batch (`None`). If users are optimised independently and stock is ignored,
the independent lists sum to `6.2964`, but they oversell item 1 (needed by
two users, stock 1) and would not be executable.

Then **only the training evidence changes** (shallower cascade; refit gives
`s ≈ [−0.356, 1.652]`), while features, `allowed`, stock and category limits
stay fixed. The value tensor is rebuilt from the new `w, s`, and the joint
allocator is rerun. The optimal lists change completely, e.g. user 0 goes
`[3,7,2] → [3,5,2]`, and the total rises to `7.3667`. The reported tensor and
lists always come from the current parameters.

## 4. Verification

`python3 verify.py` runs 209 checks: finite-difference gradients for `P=1`,
`P=3` with repeated rows and multiple positions, near-box boundary points,
probability semantics (`p = e·r`), stopping status and residual use of the
box projection, MILP-vs-enumeration agreement, legal-list re-evaluation,
`P>M` infeasibility, and the greedy failure. Everything must pass for the
script to exit zero.

## 5. Input validation

Shapes, indices, probabilities, costs, parameter boxes and finiteness are
validated and raise `ValueError` (or `InputError`, a subclass) rather than
silently clipping or coercing. Examples: feature dimension outside 1..4,
feature outside `[-1,1]`, non-binary click, non-positive/non-integer weight,
position outside `0..P−1`, ridge outside `[0.01,1]`, parameter outside
`[-3,3]`, value tensor outside `[0,1]`, non-finite entries, wrong shapes,
`stock` outside `0..U`, category outside 0..2, `min > max` or bounds outside
`0..P`.
