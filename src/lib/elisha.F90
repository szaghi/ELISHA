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
implicit none
private
public :: I4P, I8P, R8P, R_P
endmodule elisha
