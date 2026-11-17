"""Position-aware click learning and shared-stock slate allocation.

Public modules:

* :mod:`starter.click_model` -- ``fit``, ``objective``, ``probabilities``
  (relevance ``r``, examination ``e`` and click probability ``p``).
* :mod:`starter.allocation` -- ``allocate`` and helpers for an arbitrary legal
  ``U x M x P`` value tensor.
* :mod:`starter.slates` -- the original independent recommender plus
  ``recommend_slates`` built on the joint allocator.
"""