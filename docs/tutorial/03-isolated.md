---
title: 3. A star in empty space
---

# 3. A star in empty space

A star's potential goes to zero far from it; a periodic box has no "far", only the next star of an infinite lattice
of copies. **Isolated** boundaries solve the free-space problem: $L_h\phi = 4\pi G\rho$ with $\phi \to 0$ at infinity,
the box being a window on empty space. This chapter puts a Gaussian star at the centre of the unit cube,
$\rho = e^{-r^2/2\sigma^2}$, $\sigma = 0.07$, and compares three solves with its analytic potential
$-M\,\mathrm{erf}(r/\sqrt2\sigma)/r$ ($G = 1$).

## The mesh

Isolated boundaries need every axis non-periodic and cubic cells; they take no `bc`.

<<< @/examples/snippets/t03_isolated-mesh.F90{fortran}

<<< @/examples/snippets/t03_isolated-source.F90{fortran}

## Two paths to free space

`isolated=.true.` at setup. On a uniform mesh the facade chooses the spectral path, which does Hockney and
Eastwood's trick: the box is padded with zeros to at least twice its size, and the source is convolved with the
free-space Green's function of the *discrete* Laplacian (the lattice Green's function, tabulated in `elisha_lattice_green`) by FFT. The hierarchical
path, forced here with `path=ELISHA_PATH_HIERARCHICAL`, does James's: a multigrid solve with zero boundary values,
the screening charge it induces on the boundary, that charge's free-space potential on the boundary, and a second
multigrid solve with those values.

<<< @/examples/snippets/t03_isolated-solve.F90{fortran}

And for comparison the same star in a periodic box:

<<< @/examples/snippets/t03_isolated-periodic.F90{fortran}

## Running it

<<< @/examples/snippets/t03_isolated-compare.F90{fortran}

<<< @/examples/output/t03_isolated.txt{text}

- **Both isolated paths give the same potential.** The program checks 1e-10 of the central potential; they agree
  far closer, at the multigrid's tolerance (1e-12 here). Not a coincidence: each solves the same discrete
  free-space problem, by different algebra.
- **Their error against the analytic potential is 0.9%**, the discretisation's, at 32³ cells and 2.2 cells per
  $\sigma$ ([refine where it matters](/tutorial/04-amr) to do better).
- **The periodic potential is off by 24%.** It is the potential of the star *and its images*, with the mean
  density removed: shifted by a constant and flattened at the box's edge, where the images pull back.

![left: phi along the x axis through the centre, the analytic curve, the isolated solution on it and the periodic
one well above it; right: the isolated potential on the plane z = 1/2, concentric rings](/tutorial/t03-isolated.png)

::: tip What you learned
`isolated=.true.` and a non-periodic mesh of cubic cells give the free-space potential, on either path. A periodic
solve of an isolated object is the potential of a lattice of copies; the difference is not small.
:::

<details><summary>The whole program</summary>

<<< @/examples/snippets/t03_isolated.F90{fortran}

</details>

Next: [4. Refine where it matters](/tutorial/04-amr).
