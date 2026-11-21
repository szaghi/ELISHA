---
title: 6. Watching the FCG
---

# 6. Watching the FCG

The hierarchical path is iterative: flexible conjugate gradients (FCG), each iteration preconditioned by one
multigrid V-cycle, until the largest residual falls below `tolerance` times the largest source. This chapter stops
it on purpose after $k$ iterations, $k = 1, 2, \ldots$, to watch it converge, and to meet the status it returns when
it has not.

The problem: two embers (Gaussians) on a periodic 2-D mesh (z a null axis), refined twice around them.

<<< @/examples/snippets/t06_iterations-mesh.F90{fortran}

## Stopping it early

`max_iterations` caps the FCG. A solve that reaches it above its tolerance returns
**`ELISHA_ERR_NOT_CONVERGED`**: not a failure like the others, because `phi` holds the last iterate, its ghosts
filled, and `diagnostics()` says how far it got. Every rank returns the same status.

<<< @/examples/snippets/t06_iterations-history.F90{fortran}

Each $k$ gets a fresh setup: a hierarchical solve starts from the previous solution of the same setup (a warm start),
and here every solve must start from zero.

## Two V-cycles

The loop runs twice: with the default V-cycle, `ELISHA_VCYCLE_SYMMETRIC`, and with ADAM's, `ELISHA_VCYCLE_ADAM`.
They differ at the 2:1 seams. ADAM's V-cycle couples a level to the coarser one across the seam on the way up and
never applies the adjoint of that coupling on the way down, so the preconditioner is not symmetric; FCG's
convergence theory assumes it is. ELISHA's applies the adjoint, and the preconditioner is symmetric to round-off
([the hierarchical solver](/reference/hierarchical#the-symmetric-v-cycle)).

<<< @/examples/output/t06_iterations.txt{text}

![the residual against the FCG iteration on a log scale: the symmetric V-cycle reaches 1e-10 at iteration 12, ADAM's
at 19](/tutorial/t06-history.png)

The first iteration leaves the largest residual about where it started (1.0 to two digits): FCG minimises the error
in the energy norm, not the largest residual. From there the symmetric V-cycle gains about a decade per iteration and converges in 12; ADAM's
needs 19. On this mesh ADAM's V-cycle is merely slower. On another (a 2-D Neumann/Dirichlet AMR case of the
verification) it stalled at 200 iterations with the residual at 0.12, which is how the asymmetry was found.

The error of each iterate against the converged potential, on a log scale (the ground is $10^{-10}$, the tolerance):

<div class="elisha-casts">
<figure class="elisha-cast elisha-cast-wide" style="--ar: 1.67">
<video class="light-only" autoplay loop muted playsinline poster="/tutorial/casts/t06-iterates-light.png" aria-label="The error of the FCG iterates, both V-cycles"><source src="/tutorial/casts/t06-iterates-light.webm" type="video/webm"><source src="/tutorial/casts/t06-iterates-light.mp4" type="video/mp4"></video>
<video class="dark-only" autoplay loop muted playsinline poster="/tutorial/casts/t06-iterates-dark.png" aria-label="The error of the FCG iterates, both V-cycles"><source src="/tutorial/casts/t06-iterates-dark.webm" type="video/webm"><source src="/tutorial/casts/t06-iterates-dark.mp4" type="video/mp4"></video>
<figcaption>The k-th FCG iterate of each V-cycle, its error against the converged solution: the symmetric V-cycle
cools to the ground in 12 iterations, ADAM's in 19. The slower iterates are not wrong at the seams only: their error
is spread over the whole mesh.</figcaption>
</figure>
</div>

## What to do with `ELISHA_ERR_NOT_CONVERGED`

It is a status like any other: the call returns, and the caller decides. A time-dependent solver can take the last
iterate and go on (ADAM did, after printing a warning); a verification run should stop. The
[cookbook](/cookbook/not-converged) has the pattern. Keep `ELISHA_VCYCLE_ADAM` for one purpose: checking results
against ADAM's, bit for bit; then switch to the default.

::: tip What you learned
`max_iterations` and `tolerance` bound the FCG; `ELISHA_ERR_NOT_CONVERGED` comes with a usable `phi` and the
diagnostics. The default V-cycle is symmetric; ADAM's is kept for parity: it was slower on every AMR mesh measured,
and stalled on one.
:::

<details><summary>The whole program</summary>

<<< @/examples/snippets/t06_iterations.F90{fortran}

</details>

Next: [7. Many ranks](/tutorial/07-mpi).
