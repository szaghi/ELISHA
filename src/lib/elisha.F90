!< ELISHA, Elliptic Laplacian Inversion: Spectral and Hierarchical Algorithms: the library's entry point.
module elisha
!< ELISHA, Elliptic Laplacian Inversion: Spectral and Hierarchical Algorithms: the library's entry point.
!<
!< Seed of the repository: it re-exports the working kinds, so a client needs `use elisha` only. The elliptic solvers
!< extracted from ADAM's FLAIL are added behind this module: the spectral (FFT) solver on uniform block lattices and
!< the hierarchical (multigrid-preconditioned flexible conjugate gradients) solver on 2:1-balanced octrees of blocks,
!< behind one facade, distributed over a caller-given MPI communicator (or serial, `-DELISHA_NO_MPI`).
use penf, only : I4P, I8P, R8P
use elisha_kind_parameters
use elisha_parameters
use elisha_spectral
use elisha_extension
use elisha_lattice_green_library
implicit none
private
public :: I4P, I8P, R8P, R_P
! boundary kinds (elisha_parameters)
public :: ELISHA_BC_PERIODIC, ELISHA_BC_DIRICHLET, ELISHA_BC_NEUMANN, ELISHA_BC_DIRICHLET_GHOST
public :: ELISHA_BC_ZERO_GHOSTS, ELISHA_BC_FREE_GHOSTS, ELISHA_BC_KINDS
! L0 spectral math (elisha_spectral)
public :: ELISHA_D1_COEF, elisha_axis_eigenvalue, elisha_axis_transform_size, elisha_dg_stencil, elisha_smooth_size
! L0 extension maps (elisha_extension)
public :: elisha_axis_map, elisha_set_plane
! lattice Green's function of the 7-point Laplacian (elisha_lattice_green_library, generated)
public :: elisha_lattice_green
endmodule elisha
