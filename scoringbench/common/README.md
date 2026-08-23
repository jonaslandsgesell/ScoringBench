# `scoringbench.common`

**Intentionally empty for now.**

This package exists as a placeholder only. It will be populated in a later phase,
after the two self-contained subpackages — `scoringbench.univariate` and
`scoringbench.multivariate` — are both in place and the *true* shared surface
between them has been observed in practice.

## Why empty on purpose

The restructure adopts "Alternative A, pure form": two **self-contained,
symmetric** subpackages with **no import path between them in either
direction**. Prematurely hoisting code into `common/` would recreate exactly the
coupling this split exists to remove, and would force guesses about which
abstractions are genuinely shared before we have evidence.

Therefore, at this stage:

- **Nothing is moved or hoisted into `common/`.**
- **Duplication between `univariate/` and `multivariate/` is ACCEPTED and
  EXPECTED.** It is cheaper to de-duplicate later, guided by real usage, than to
  design a shared layer up front and get it wrong.
