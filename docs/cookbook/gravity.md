---
title: Self-gravity
---

# Self-gravity

The gravitational potential of a density $\rho$ solves $\nabla^2\Phi = 4\pi G\rho$, with $\Phi \to 0$ far away for an
isolated system. In ELISHA's terms: the source is $\rho$, `factor` is $4\pi G$, and the boundaries are `isolated`.
The optional `energy` returns $\sum \rho\,\Phi\,\Delta V$ over the cells, on every rank, which is twice the
gravitational binding energy $W = \tfrac12\int\rho\Phi\,dV$.

<<< @/examples/snippets/cookbook_gravity-gravity.F90{fortran}

A Gaussian cloud of mass $M$ and width $\sigma$ has $W = -GM^2/(2\sqrt\pi\,\sigma)$. The program measures it on 16³
and 32³ cells:

<<< @/examples/output/cookbook_gravity.txt{text}

Second order: the error falls by 4.1 per halving of the cell.

- **Units**: `factor` carries $G$ in your units (cgs here); ELISHA has no units of its own.
- **A periodic box** (cosmology) solves $\nabla^2\Phi = 4\pi G(\rho - \bar\rho)$: the mean is removed for you
  ([Your fields in and out](/cookbook/fields)).
- **On AMR** the energy sums over every leaf with its own cell volume: nothing changes in the call.
- **Where the force comes from**: the potential comes back with its ghosts filled, so $-\nabla\Phi$ by central
  differences needs no exchange of your own on the face ghosts ([the solver facade](/reference/solver#solve)).

<details><summary>The whole program</summary>

<<< @/examples/snippets/cookbook_gravity.F90{fortran}

</details>
