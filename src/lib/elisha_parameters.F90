!< ELISHA, the shared parameters: the boundary kinds of a domain face.
module elisha_parameters
!< ELISHA, the shared parameters: the boundary kinds of a domain face.
!<
!< One kind per face, `bc(2,3)` = [lo/hi, axis], shared by the spectral solver, the hierarchical solver and the facade.
!< The values are ADAM's `FLAIL_BC_*` (`adam_flail_parameters`), so an adapter passes them through unchanged.
!<   - `ELISHA_BC_PERIODIC`: both faces of the axis.
!<   - `ELISHA_BC_DIRICHLET`: the value at the **face** (homogeneous: `phi_ghost = -phi_boundary`; the wall at the face,
!<     second order). The spectral solver extends the source odd about the face.
!<   - `ELISHA_BC_NEUMANN`: the normal derivative at the face (homogeneous: `phi_ghost = phi_boundary`). The spectral
!<     solver extends the source even about the face. Without a Dirichlet face anywhere the operator is singular and the
!<     solve drops the source mean, as on a periodic grid.
!<   - `ELISHA_BC_DIRICHLET_GHOST`: the value at the centre of the first ghost cell (the hierarchical solver's root and
!<     its James step, where the ghost values are computed, not prescribed). The wall sits half a cell outside the
!<     domain: as a prescribed condition it is first order.
!<   - `ELISHA_BC_ZERO_GHOSTS`: every ghost layer zero (PRISM's elliptic Dirichlet face). With the compact operator it
!<     is `ELISHA_BC_DIRICHLET_GHOST`; with the wide D(G) it is no reflection, and the spectral solver solves it by
!<     preconditioned CG.
!<   - `ELISHA_BC_FREE_GHOSTS`: every ghost layer holds the continuum free-space potential of the source (PRISM's
!<     open face), with the wide D(G) only.
use penf, only : I4P
implicit none
private
public :: ELISHA_BC_PERIODIC
public :: ELISHA_BC_DIRICHLET
public :: ELISHA_BC_NEUMANN
public :: ELISHA_BC_DIRICHLET_GHOST
public :: ELISHA_BC_ZERO_GHOSTS
public :: ELISHA_BC_FREE_GHOSTS
public :: ELISHA_BC_KINDS

integer(I4P), parameter :: ELISHA_BC_PERIODIC        = 0_I4P !< Periodic (both faces of the axis).
integer(I4P), parameter :: ELISHA_BC_DIRICHLET       = 1_I4P !< Value at the face.
integer(I4P), parameter :: ELISHA_BC_NEUMANN         = 2_I4P !< Normal derivative at the face.
integer(I4P), parameter :: ELISHA_BC_DIRICHLET_GHOST = 3_I4P !< Value at the first ghost centre.
integer(I4P), parameter :: ELISHA_BC_ZERO_GHOSTS     = 4_I4P !< Every ghost layer zero (wide D(G): CG).
integer(I4P), parameter :: ELISHA_BC_FREE_GHOSTS     = 5_I4P !< Ghosts: the free-space potential (wide D(G)).
integer(I4P), parameter :: ELISHA_BC_KINDS(6)=[ELISHA_BC_PERIODIC, ELISHA_BC_DIRICHLET, ELISHA_BC_NEUMANN,      &
                                               ELISHA_BC_DIRICHLET_GHOST, ELISHA_BC_ZERO_GHOSTS,              &
                                               ELISHA_BC_FREE_GHOSTS] !< Every kind, for validation.
endmodule elisha_parameters
