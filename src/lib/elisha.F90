!< ELISHA, Elliptic Laplacian Inversion: Spectral and Hierarchical Algorithms: the library's entry point.
module elisha
!< ELISHA, Elliptic Laplacian Inversion: Spectral and Hierarchical Algorithms: the library's entry point.
!<
!< It re-exports the whole host library, so a client needs `use elisha` only (and `use elisha_device` for the device
!< backends, `*-nvf-oac` builds). The elliptic solvers extracted from ADAM's FLAIL: the spectral (FFT) solver on uniform
!< block lattices and the hierarchical (multigrid-preconditioned flexible conjugate gradients) solver on 2:1-balanced
!< octrees of blocks, distributed over a caller-given MPI communicator (or serial, `-DELISHA_NO_MPI`), and the facade
!< over both, `elisha_solver_object` (the path chosen for the mesh, `elisha_diagnostics_object`), the entry point of a
!< client. The mesh descriptor, the communication context and the status policy are what every call takes.
use penf, only : I4P, I8P, R8P
use elisha_kind_parameters
use elisha_status
use elisha_comm
use elisha_keys
use elisha_mesh
use elisha_setup_checks
use elisha_parameters
use elisha_spectral
use elisha_extension
use elisha_lattice_green_library
use elisha_spectral_solver
use elisha_hierarchical_solver
use elisha_solver
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
public :: ELISHA_ERR_NOT_CONVERGED
! the communication context and the halo plans (elisha_comm)
public :: elisha_comm_object, elisha_halo_plan
! cell keys and the hash table (elisha_keys)
public :: ELISHA_KEY_COORD_LIMIT, ELISHA_KEY_LEVEL_OFFSET, ELISHA_KEY_LEVEL_MAX
public :: elisha_bkey, elisha_bkey4, elisha_ckey, elisha_kdecode, elisha_hash_slot, elisha_hash_table
! the mesh descriptor (elisha_mesh)
public :: elisha_mesh_object
! boundary kinds (elisha_parameters)
public :: ELISHA_BC_PERIODIC, ELISHA_BC_DIRICHLET, ELISHA_BC_NEUMANN, ELISHA_BC_DIRICHLET_GHOST
public :: ELISHA_BC_ZERO_GHOSTS, ELISHA_BC_FREE_GHOSTS, ELISHA_BC_KINDS
! solve paths (elisha_parameters)
public :: ELISHA_PATH_AUTO, ELISHA_PATH_SPECTRAL, ELISHA_PATH_HIERARCHICAL
! level operators of the hierarchical solver (elisha_parameters)
public :: ELISHA_OPERATOR_CSR, ELISHA_OPERATOR_STENCIL
! V-cycles of the hierarchical solver (elisha_parameters)
public :: ELISHA_VCYCLE_SYMMETRIC, ELISHA_VCYCLE_ADAM
! where the hierarchical solver assembles a solve (elisha_parameters)
public :: ELISHA_ASSEMBLY_HOST, ELISHA_ASSEMBLY_DEVICE
! the setup checks (elisha_setup_checks)
public :: elisha_check_bc, elisha_check_dg_order, elisha_check_ghost_count, elisha_check_isolated
public :: elisha_check_isolated_mesh, elisha_check_path, elisha_check_periodicity, elisha_check_reach
public :: elisha_check_unbounded
! L0 spectral math (elisha_spectral)
public :: ELISHA_D1_COEF, elisha_axis_eigenvalue, elisha_axis_transform_size, elisha_dg_stencil, elisha_smooth_size
! L0 extension maps (elisha_extension)
public :: elisha_axis_map, elisha_set_plane
! lattice Green's function of the 7-point Laplacian (elisha_lattice_green_library, generated)
public :: elisha_lattice_green
! the spectral (FFT) solver (elisha_spectral_solver)
public :: elisha_spectral_solver_object
! the hierarchical (multigrid FCG) solver (elisha_hierarchical_solver): the type, its CSR and vector types (a device
! backend extends the hooks on them), the 2:1 seam's ghost constant (FLUME inverts it)
public :: elisha_hierarchical_solver_object, dcsr, rvec, ivec, MG_SEAM_GHOST_SCALE
! the facade and its diagnostics (elisha_solver)
public :: elisha_solver_object, elisha_diagnostics_object
endmodule elisha
