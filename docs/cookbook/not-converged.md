---
title: When it does not converge
---

# When it does not converge

The hierarchical path's FCG stops at `max_iterations` (default 200). If its residual is still above the tolerance
then, `solve_packed` returns **`ELISHA_ERR_NOT_CONVERGED`**, on every rank. So does the spectral path's CG, the solve
of PRISM's wide operator with zero or free ghosts. Unlike every other status, it comes with a result:

- `phi` is the last iterate, its ghosts filled, and `energy` is its energy;
- `diagnostics()` tells how far it got: `iterations`, `residual`, `tolerance_used`, and `converged = .false.`.

The pattern, from [tutorial 6](/tutorial/06-iterations):

<<< @/examples/snippets/t06_iterations-history.F90{fortran}

What to do is the caller's choice:

| Situation | Choice |
|---|---|
| A time step that can live with a slightly worse potential | take `phi`, log `diagnostics()`, go on (what ADAM did, with a warning on rank 0) |
| A verification or a convergence study | stop: the number you would print is not the solver's answer |
| It happens every step | read `diagnostics()`: a residual still falling at the cap needs more iterations (or a looser `tolerance`); one stuck far above it on `ELISHA_VCYCLE_ADAM` is that V-cycle's asymmetry, switch to the default |

`ELISHA_VCYCLE_ADAM`, ADAM's V-cycle, is not symmetric on AMR meshes and FCG can stall on it: one case of the
verification stalled at 200 iterations with the residual at 0.12, and converges in 15 with the default V-cycle
([the hierarchical solver](/reference/hierarchical#the-symmetric-v-cycle)). Use it only to compare with ADAM.
