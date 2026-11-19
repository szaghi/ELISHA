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
use elisha_status
use elisha_comm
use elisha_keys
use elisha_parameters
use elisha_spectral
use elisha_extension
use elisha_lattice_green_library
implicit none
private
public :: I4P, I8P, R8P, R_P
! the status policy (elisha_status): every code, the helpers
public :: ELISHA_OK, ELISHA_STATUS_LAST, elisha_set_error, elisha_status_name, elisha_status_text
public :: ELISHA_ERR_ARGUMENT, ELISHA_ERR_NO_FFT, ELISHA_ERR_FFT_PLAN, ELISHA_ERR_MPI_NOT_INITIALIZED, ELISHA_ERR_MPI
public :: ELISHA_ERR_INTERNAL, ELISHA_ERR_MESH_SHAPE, ELISHA_ERR_MESH_COORDS, ELISHA_ERR_MESH_LEVELS
public :: ELISHA_ERR_MESH_COVERAGE, ELISHA_ERR_MESH_BALANCE, ELISHA_ERR_MESH_KEY_LIMIT, ELISHA_ERR_MESH_CHANGED
public :: ELISHA_ERR_BC_UNKNOWN, ELISHA_ERR_BC_PERIODIC_PAIR, ELISHA_ERR_BC_GHOST_PAIR, ELISHA_ERR_BC_ZERO_GHOST_PAIR
public :: ELISHA_ERR_BC_PERIODICITY, ELISHA_ERR_BC_MISSING, ELISHA_ERR_BC_FREE_GHOSTS
public :: ELISHA_ERR_ISOLATED_BC, ELISHA_ERR_ISOLATED_WIDE, ELISHA_ERR_ISOLATED_DIMENSION, ELISHA_ERR_ISOLATED_PERIODIC
public :: ELISHA_ERR_ISOLATED_CELLS, ELISHA_ERR_ISOLATED_BOUNDARY_LEAF
public :: ELISHA_ERR_DG_ORDER, ELISHA_ERR_DG_REACH, ELISHA_ERR_DG_AMR, ELISHA_ERR_REDUNDANT, ELISHA_ERR_GHOST_VALUES
public :: ELISHA_ERR_GHOST_COUNT, ELISHA_ERR_NOT_INITIALIZED, ELISHA_ERR_SHAPE, ELISHA_ERR_ENERGY_REDUNDANT
! the communication context (elisha_comm)
public :: elisha_comm_object
! cell keys and the hash table (elisha_keys)
public :: ELISHA_KEY_COORD_LIMIT, ELISHA_KEY_LEVEL_OFFSET, ELISHA_KEY_LEVEL_MAX
public :: elisha_bkey, elisha_ckey, elisha_kdecode, elisha_hash_table
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
