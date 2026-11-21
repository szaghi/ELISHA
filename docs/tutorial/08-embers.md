---
title: 8. Two embers
---

# 8. Two embers

ELISHA's mark is two embers in one hearth: the rings raked in the ash around them are the contours of a Poisson solve
with two point sources, small rings around each, one ring that takes in both at the waist. This chapter makes it
move. Two Gaussian masses turn on a circle in free space, and the mesh follows them: every step is a **regrid**, a new
mesh refined around the embers' new places, and a new setup on it.

<div class="elisha-casts">
<figure class="elisha-cast" style="--ar: 1.0">
<video class="light-only" autoplay loop muted playsinline poster="/tutorial/casts/t08-embers-light.png" aria-label="Two embers orbiting on an adaptive mesh"><source src="/tutorial/casts/t08-embers-light.webm" type="video/webm"><source src="/tutorial/casts/t08-embers-light.mp4" type="video/mp4"></video>
<video class="dark-only" autoplay loop muted playsinline poster="/tutorial/casts/t08-embers-dark.png" aria-label="Two embers orbiting on an adaptive mesh"><source src="/tutorial/casts/t08-embers-dark.webm" type="video/webm"><source src="/tutorial/casts/t08-embers-dark.mp4" type="video/mp4"></video>
<figcaption>The plane z = 1/2 of each step: the potential and its rings, the block outlines darker (light theme) or
brighter (dark theme) for finer levels, the embers' centres as dots. Half a turn in 48 steps: two equal masses at the
ends of a diameter look the same after half a turn, so the loop is seamless.</figcaption>
</figure>
</div>

## A step

The embers at the ends of a diameter:

<<< @/examples/snippets/t08_embers-orbit.F90{fortran}

The regrid: the leaves around the embers' new places (the refinement of chapter 4, with reaches that keep the mesh
2:1 and the boundary at level 0), a new mesh descriptor, and `setup` on it. `setup` destroys the solver it held and
builds the one the new mesh needs; it could as well switch paths, if a step's mesh were uniform.

<<< @/examples/snippets/t08_embers-regrid.F90{fortran}

<<< @/examples/snippets/t08_embers-solve.F90{fortran}

## Running it

The docs run solves every 8th step:

<<< @/examples/output/t08_embers.txt{text}

- **The mesh breathes with the angle.** The embers move through a lattice of blocks fixed in space, so the number of
  refined blocks changes as they cross block boundaries: 240 leaves at levels 1 and 2 when the embers lie along an
  axis, 304 at 30° and 60°.
- **The FCG count barely moves**: 14 or 15, whatever the mesh under the embers.
- **The energy is constant to three digits.** Turning the pair changes nothing physical; the fourth digit is the mesh
  changing under it.

## The cost of a regrid

`setup` is not free: it gathers and validates the leaf list, builds the V-cycle's levels and operators, and plans the
root's FFT. On the 3.9 M-cell AMR mesh of the scaling ladder, one rank, it takes 15.6 s against 5.88 s for a solve
([Performance](/guide/performance#scaling-eb)). A solver whose mesh changes every step pays it every step. One whose mesh changes every few steps solves
on the same setup in between, and each of those solves starts from the previous potential
([One setup, many solves](/cookbook/reuse)).

::: tip What you learned
A regrid is a new mesh descriptor and a new `setup`; the facade chooses the path again. A hierarchical setup can
cost more than a solve: reuse it while the mesh does not change.
:::

<details><summary>The whole program</summary>

<<< @/examples/snippets/t08_embers.F90{fortran}

</details>

Next: [9. On the GPU](/tutorial/09-gpu).
