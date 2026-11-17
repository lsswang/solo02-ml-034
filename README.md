# Click learning and shared recommendation allocation

Run `python3 demo.py` using Python 3.11, NumPy and SciPy. Current click_model fits a relevance-only logistic model. slates recommends for each user independently. Both are working ordinary tools. Synthetic training rows aggregate repeated binary events using positive integer weights. Position indices are zero-based.
The expanded task defines a monotone examination model and shared-stock slot allocation. External allocation can consume an arbitrary legal probability tensor without fitting this model. No network, real user interaction or online service is required.

## What is implemented

* `starter/click_model.py`: position-aware click model
  `r(x) = sigmoid(x·w)`, `e[0] = 1`,
  `e[j] = Π_{k=1..j} sigmoid(s[k-1])`, `p(x,j) = e[j]·r(x)`.
  `fit` minimises the weighted negative log likelihood plus
  `ridge/2·(||w||²+||s||²)` inside `[-3,3]` and reports the true objective,
  the full `[w, s]` gradient, the projected-gradient residual and an explicit
  stop status. `objective` returns `(value, gradient)`; `probabilities`
  returns `r`, `e` and `p` separately.
* `starter/allocation.py`: one joint MILP over a `U×M×P` value tensor with
  per-user allowed items, no repeated item per user, shared stock, and
  per-category lower/upper bounds. `status` is `optimal` or `infeasible`;
  the reported value is recomputed from the chosen list.
  An exact complete-enumeration solver is available via `method="enumerate"`
  and is used as the independent check.
* `starter/slates.py`: the original independent greedy recommender plus
  `recommend_slates`, which routes to the joint allocator.
* `verify.py`: finite-difference gradients (`P=1`, repeated rows, several
  display positions, box boundaries), probability semantics, stopping
  status, and MILP-vs-brute-force allocation checks including a genuine
  infeasible case and a naive greedy failure.
* `starter/experiments.py`: synthetic study that learns the model, builds the
  value tensor from `e[j]·sigmoid(x·w)`, allocates the batch, then changes
  only the training evidence, refits, rebuilds the tensor and reallocates.

## Results

Run `python3 verify.py` for the checks and `python3 -m starter.experiments`
for the synthetic study. See `NOTES.md` for the method summary and measured
numbers.

## Environment

If NumPy/SciPy are not installed system-wide, use a virtual environment:

```
python3 -m venv .venv
.venv/bin/pip install numpy scipy
.venv/bin/python demo.py
```
