!< ELISHA, the status policy: a code and a message, never a stop.
module elisha_status
!< ELISHA, the status policy: a code and a message, never a stop.
!<
!< Every procedure of ELISHA that can fail returns `stat` (mandatory, `intent(out)`): `ELISHA_OK`, or one code per
!< class of failure, and optionally `errmsg` (`character(:), allocatable`), a sentence naming what failed and the values
!< involved. The library never executes `error stop` (owner, 2026-11-19): the caller decides. Device code, which cannot
!< return a status, writes NaN instead (GIDEON's convention).
!<
!< The codes come from ADAM's refusals (`error_stop` in FLAIL, ADAM 669d0bdd): (A) the setup checks of the arguments,
!< (B) the collective checks of the mesh, (C) runtime and internal failures; `elisha_status_name(code)` is the
!< parameter's name, `elisha_status_text(code)` its one-line meaning. One code is not ADAM's: an iterative solve (the
!< hierarchical solver's FCG, the spectral solver's CG of the wide D(G)) that reaches `max_iterations` above its
!< tolerance returns `ELISHA_ERR_NOT_CONVERGED`, every rank alike, with `phi` the last iterate (ghosts filled), where
!< ADAM printed a warning on rank 0 and returned as if converged (ELISHA notes E1c, decision C).
use penf, only : I4P
implicit none
private
public :: elisha_set_error, elisha_status_name, elisha_status_text
public :: ELISHA_STATUS_LAST
public :: ELISHA_OK
public :: ELISHA_ERR_ARGUMENT
public :: ELISHA_ERR_NO_FFT
public :: ELISHA_ERR_FFT_PLAN
public :: ELISHA_ERR_MPI_NOT_INITIALIZED
public :: ELISHA_ERR_MPI
public :: ELISHA_ERR_INTERNAL
public :: ELISHA_ERR_MESH_SHAPE
public :: ELISHA_ERR_MESH_COORDS
public :: ELISHA_ERR_MESH_LEVELS
public :: ELISHA_ERR_MESH_COVERAGE
public :: ELISHA_ERR_MESH_BALANCE
public :: ELISHA_ERR_MESH_KEY_LIMIT
public :: ELISHA_ERR_MESH_CHANGED
public :: ELISHA_ERR_BC_UNKNOWN
public :: ELISHA_ERR_BC_PERIODIC_PAIR
public :: ELISHA_ERR_BC_GHOST_PAIR
public :: ELISHA_ERR_BC_ZERO_GHOST_PAIR
public :: ELISHA_ERR_BC_PERIODICITY
public :: ELISHA_ERR_BC_MISSING
public :: ELISHA_ERR_BC_FREE_GHOSTS
public :: ELISHA_ERR_ISOLATED_BC
public :: ELISHA_ERR_ISOLATED_WIDE
public :: ELISHA_ERR_ISOLATED_DIMENSION
public :: ELISHA_ERR_ISOLATED_PERIODIC
public :: ELISHA_ERR_ISOLATED_CELLS
public :: ELISHA_ERR_ISOLATED_BOUNDARY_LEAF
public :: ELISHA_ERR_DG_ORDER
public :: ELISHA_ERR_DG_REACH
public :: ELISHA_ERR_DG_AMR
public :: ELISHA_ERR_REDUNDANT
public :: ELISHA_ERR_GHOST_VALUES
public :: ELISHA_ERR_GHOST_COUNT
public :: ELISHA_ERR_NOT_INITIALIZED
public :: ELISHA_ERR_SHAPE
public :: ELISHA_ERR_ENERGY_REDUNDANT
public :: ELISHA_ERR_NOT_CONVERGED

integer(I4P), parameter :: ELISHA_OK = 0_I4P                     !< Success.
integer(I4P), parameter :: ELISHA_ERR_ARGUMENT = 1_I4P           !< An argument is out of its range.
integer(I4P), parameter :: ELISHA_ERR_NO_FFT = 2_I4P             !< Built without an FFT library.
integer(I4P), parameter :: ELISHA_ERR_FFT_PLAN = 3_I4P           !< The FFT library could not make or execute a plan.
integer(I4P), parameter :: ELISHA_ERR_MPI_NOT_INITIALIZED = 4_I4P !< MPI is not initialised.
integer(I4P), parameter :: ELISHA_ERR_MPI = 5_I4P                !< An MPI call failed.
integer(I4P), parameter :: ELISHA_ERR_INTERNAL = 6_I4P           !< An internal invariant failed (a bug in ELISHA).
integer(I4P), parameter :: ELISHA_ERR_MESH_SHAPE = 7_I4P         !< The mesh shape is invalid.
integer(I4P), parameter :: ELISHA_ERR_MESH_COORDS = 8_I4P        !< A block coordinate is outside its level's range.
integer(I4P), parameter :: ELISHA_ERR_MESH_LEVELS = 9_I4P        !< The blocks sit at more than one level where one is.
integer(I4P), parameter :: ELISHA_ERR_MESH_COVERAGE = 10_I4P     !< The blocks do not cover the domain exactly.
integer(I4P), parameter :: ELISHA_ERR_MESH_BALANCE = 11_I4P      !< The mesh is not 2:1 balanced across faces.
integer(I4P), parameter :: ELISHA_ERR_MESH_KEY_LIMIT = 12_I4P    !< The finest level has 2^19 or more cells along an.
integer(I4P), parameter :: ELISHA_ERR_MESH_CHANGED = 13_I4P      !< The mesh does not match the setup.
integer(I4P), parameter :: ELISHA_ERR_BC_UNKNOWN = 14_I4P        !< An unknown boundary kind.
integer(I4P), parameter :: ELISHA_ERR_BC_PERIODIC_PAIR = 15_I4P  !< An axis has one periodic face and one non-periodic.
integer(I4P), parameter :: ELISHA_ERR_BC_GHOST_PAIR = 16_I4P     !< An axis mixes a ghost-Dirichlet face with another.
integer(I4P), parameter :: ELISHA_ERR_BC_ZERO_GHOST_PAIR = 17_I4P !< An axis mixes zero/free ghosts with another kind.
integer(I4P), parameter :: ELISHA_ERR_BC_PERIODICITY = 18_I4P    !< The boundary kinds disagree with the mesh's.
integer(I4P), parameter :: ELISHA_ERR_BC_MISSING = 19_I4P        !< A non-periodic axis without boundary kinds.
integer(I4P), parameter :: ELISHA_ERR_BC_FREE_GHOSTS = 20_I4P    !< Free-space ghosts need the wide operator D.
integer(I4P), parameter :: ELISHA_ERR_ISOLATED_BC = 21_I4P       !< Isolated boundaries take no boundary kinds.
integer(I4P), parameter :: ELISHA_ERR_ISOLATED_WIDE = 22_I4P     !< Isolated boundaries take the compact operator only.
integer(I4P), parameter :: ELISHA_ERR_ISOLATED_DIMENSION = 23_I4P !< Isolated boundaries are 3-D only.
integer(I4P), parameter :: ELISHA_ERR_ISOLATED_PERIODIC = 24_I4P !< Isolated boundaries need every axis non-periodic.
integer(I4P), parameter :: ELISHA_ERR_ISOLATED_CELLS = 25_I4P    !< Isolated boundaries need cubic cells.
integer(I4P), parameter :: ELISHA_ERR_ISOLATED_BOUNDARY_LEAF = 26_I4P !< Isolated boundaries need every boundary leaf.
integer(I4P), parameter :: ELISHA_ERR_DG_ORDER = 27_I4P          !< The order of the wide operator is not 0, 2, 4, 6 or.
integer(I4P), parameter :: ELISHA_ERR_DG_REACH = 28_I4P          !< An axis has fewer cells than the wide operator's.
integer(I4P), parameter :: ELISHA_ERR_DG_AMR = 29_I4P            !< The wide operator needs a uniform mesh (spectral path).
integer(I4P), parameter :: ELISHA_ERR_REDUNDANT = 30_I4P         !< The redundant spectral solve is for the compact,.
integer(I4P), parameter :: ELISHA_ERR_GHOST_VALUES = 31_I4P      !< Boundary values need the hierarchical path and.
integer(I4P), parameter :: ELISHA_ERR_GHOST_COUNT = 32_I4P       !< Fewer boundary values than boundary faces.
integer(I4P), parameter :: ELISHA_ERR_NOT_INITIALIZED = 33_I4P   !< The solver is used before its setup.
integer(I4P), parameter :: ELISHA_ERR_SHAPE = 34_I4P             !< An array argument has the wrong shape or size.
integer(I4P), parameter :: ELISHA_ERR_ENERGY_REDUNDANT = 35_I4P  !< No energy with the redundant spectral solve.
integer(I4P), parameter :: ELISHA_ERR_NOT_CONVERGED = 36_I4P     !< An iterative solve stopped above its tolerance.
integer(I4P), parameter :: ELISHA_STATUS_LAST = 36_I4P          !< The largest code.

contains
   pure subroutine elisha_set_error(stat, errmsg, code, msg)
   !< Set the status to `code` and, when the caller asked for it, the message: `<name>: <msg>`.
   integer(I4P),                       intent(out)           :: stat   !< Status.
   character(:), allocatable,          intent(out), optional :: errmsg !< Message.
   integer(I4P),                       intent(in)            :: code   !< Code.
   character(*),                       intent(in)            :: msg    !< What failed, with its values.

   stat = code
   if (present(errmsg)) errmsg = elisha_status_name(code)//': '//msg
   endsubroutine elisha_set_error

   pure function elisha_status_name(code) result(name)
   !< The parameter's name of a code (`ELISHA_UNKNOWN_STATUS` for none).
   integer(I4P), intent(in)  :: code !< Code.
   character(:), allocatable :: name !< Name.

   select case(code)
   case(ELISHA_OK) ; name = 'ELISHA_OK'
   case(ELISHA_ERR_ARGUMENT) ; name = 'ELISHA_ERR_ARGUMENT'
   case(ELISHA_ERR_NO_FFT) ; name = 'ELISHA_ERR_NO_FFT'
   case(ELISHA_ERR_FFT_PLAN) ; name = 'ELISHA_ERR_FFT_PLAN'
   case(ELISHA_ERR_MPI_NOT_INITIALIZED) ; name = 'ELISHA_ERR_MPI_NOT_INITIALIZED'
   case(ELISHA_ERR_MPI) ; name = 'ELISHA_ERR_MPI'
   case(ELISHA_ERR_INTERNAL) ; name = 'ELISHA_ERR_INTERNAL'
   case(ELISHA_ERR_MESH_SHAPE) ; name = 'ELISHA_ERR_MESH_SHAPE'
   case(ELISHA_ERR_MESH_COORDS) ; name = 'ELISHA_ERR_MESH_COORDS'
   case(ELISHA_ERR_MESH_LEVELS) ; name = 'ELISHA_ERR_MESH_LEVELS'
   case(ELISHA_ERR_MESH_COVERAGE) ; name = 'ELISHA_ERR_MESH_COVERAGE'
   case(ELISHA_ERR_MESH_BALANCE) ; name = 'ELISHA_ERR_MESH_BALANCE'
   case(ELISHA_ERR_MESH_KEY_LIMIT) ; name = 'ELISHA_ERR_MESH_KEY_LIMIT'
   case(ELISHA_ERR_MESH_CHANGED) ; name = 'ELISHA_ERR_MESH_CHANGED'
   case(ELISHA_ERR_BC_UNKNOWN) ; name = 'ELISHA_ERR_BC_UNKNOWN'
   case(ELISHA_ERR_BC_PERIODIC_PAIR) ; name = 'ELISHA_ERR_BC_PERIODIC_PAIR'
   case(ELISHA_ERR_BC_GHOST_PAIR) ; name = 'ELISHA_ERR_BC_GHOST_PAIR'
   case(ELISHA_ERR_BC_ZERO_GHOST_PAIR) ; name = 'ELISHA_ERR_BC_ZERO_GHOST_PAIR'
   case(ELISHA_ERR_BC_PERIODICITY) ; name = 'ELISHA_ERR_BC_PERIODICITY'
   case(ELISHA_ERR_BC_MISSING) ; name = 'ELISHA_ERR_BC_MISSING'
   case(ELISHA_ERR_BC_FREE_GHOSTS) ; name = 'ELISHA_ERR_BC_FREE_GHOSTS'
   case(ELISHA_ERR_ISOLATED_BC) ; name = 'ELISHA_ERR_ISOLATED_BC'
   case(ELISHA_ERR_ISOLATED_WIDE) ; name = 'ELISHA_ERR_ISOLATED_WIDE'
   case(ELISHA_ERR_ISOLATED_DIMENSION) ; name = 'ELISHA_ERR_ISOLATED_DIMENSION'
   case(ELISHA_ERR_ISOLATED_PERIODIC) ; name = 'ELISHA_ERR_ISOLATED_PERIODIC'
   case(ELISHA_ERR_ISOLATED_CELLS) ; name = 'ELISHA_ERR_ISOLATED_CELLS'
   case(ELISHA_ERR_ISOLATED_BOUNDARY_LEAF) ; name = 'ELISHA_ERR_ISOLATED_BOUNDARY_LEAF'
   case(ELISHA_ERR_DG_ORDER) ; name = 'ELISHA_ERR_DG_ORDER'
   case(ELISHA_ERR_DG_REACH) ; name = 'ELISHA_ERR_DG_REACH'
   case(ELISHA_ERR_DG_AMR) ; name = 'ELISHA_ERR_DG_AMR'
   case(ELISHA_ERR_REDUNDANT) ; name = 'ELISHA_ERR_REDUNDANT'
   case(ELISHA_ERR_GHOST_VALUES) ; name = 'ELISHA_ERR_GHOST_VALUES'
   case(ELISHA_ERR_GHOST_COUNT) ; name = 'ELISHA_ERR_GHOST_COUNT'
   case(ELISHA_ERR_NOT_INITIALIZED) ; name = 'ELISHA_ERR_NOT_INITIALIZED'
   case(ELISHA_ERR_SHAPE) ; name = 'ELISHA_ERR_SHAPE'
   case(ELISHA_ERR_ENERGY_REDUNDANT) ; name = 'ELISHA_ERR_ENERGY_REDUNDANT'
   case(ELISHA_ERR_NOT_CONVERGED) ; name = 'ELISHA_ERR_NOT_CONVERGED'
   case default ; name = 'ELISHA_UNKNOWN_STATUS'
   endselect
   endfunction elisha_status_name

   pure function elisha_status_text(code) result(text)
   !< The one-line meaning of a code.
   integer(I4P), intent(in)  :: code !< Code.
   character(:), allocatable :: text !< Meaning.

   select case(code)
   case(ELISHA_OK) ; text = 'success'
   case(ELISHA_ERR_ARGUMENT) ; text = 'an argument is out of its range'
   case(ELISHA_ERR_NO_FFT) ; text = 'built without an FFT library (-D_FFTW on the host, -D_CUFFT on NVIDIA)'
   case(ELISHA_ERR_FFT_PLAN) ; text = 'the FFT library could not make or execute a plan'
   case(ELISHA_ERR_MPI_NOT_INITIALIZED) ; text = 'MPI is not initialised (the caller calls MPI_Init before ELISHA)'
   case(ELISHA_ERR_MPI) ; text = 'an MPI call failed'
   case(ELISHA_ERR_INTERNAL) ; text = 'an internal invariant failed (a bug in ELISHA)'
   case(ELISHA_ERR_MESH_SHAPE) ; text = 'the mesh shape is invalid (cells per block, ghosts, blocks at level 0,' // &
      ' spacing, null axes)'
   case(ELISHA_ERR_MESH_COORDS) ; text = 'a block coordinate is outside the range of its level'
   case(ELISHA_ERR_MESH_LEVELS) ; text = 'the blocks sit at more than one level where one is needed (the' // &
      ' spectral solver)'
   case(ELISHA_ERR_MESH_COVERAGE) ; text = 'the blocks do not cover the domain exactly (a gap, an overlap or a' // &
      ' duplicate)'
   case(ELISHA_ERR_MESH_BALANCE) ; text = 'the mesh is not 2:1 balanced across faces'
   case(ELISHA_ERR_MESH_KEY_LIMIT) ; text = 'the finest level has 2^19 or more cells along an axis (the' // &
      ' cell-key limit)'
   case(ELISHA_ERR_MESH_CHANGED) ; text = 'the mesh does not match the setup (a regrid needs a new setup)'
   case(ELISHA_ERR_BC_UNKNOWN) ; text = 'an unknown boundary kind'
   case(ELISHA_ERR_BC_PERIODIC_PAIR) ; text = 'an axis has one periodic face and one non-periodic face'
   case(ELISHA_ERR_BC_GHOST_PAIR) ; text = 'an axis mixes a ghost-Dirichlet face with another kind'
   case(ELISHA_ERR_BC_ZERO_GHOST_PAIR) ; text = 'an axis mixes a zero-ghost or free-ghost face with another kind'
   case(ELISHA_ERR_BC_PERIODICITY) ; text = 'the boundary kinds disagree with the periodic axes of the mesh'
   case(ELISHA_ERR_BC_MISSING) ; text = 'a non-periodic axis without boundary kinds'
   case(ELISHA_ERR_BC_FREE_GHOSTS) ; text = 'free-space ghosts need the wide operator D(G) on the spectral path'
   case(ELISHA_ERR_ISOLATED_BC) ; text = 'isolated boundaries take no boundary kinds'
   case(ELISHA_ERR_ISOLATED_WIDE) ; text = 'isolated boundaries take the compact operator only'
   case(ELISHA_ERR_ISOLATED_DIMENSION) ; text = 'isolated boundaries are 3-D only'
   case(ELISHA_ERR_ISOLATED_PERIODIC) ; text = 'isolated boundaries need every axis non-periodic'
   case(ELISHA_ERR_ISOLATED_CELLS) ; text = 'isolated boundaries need cubic cells'
   case(ELISHA_ERR_ISOLATED_BOUNDARY_LEAF) ; text = 'isolated boundaries need every boundary leaf at the' // &
      ' coarsest level'
   case(ELISHA_ERR_DG_ORDER) ; text = 'the order of the wide operator is not 0, 2, 4, 6 or 8'
   case(ELISHA_ERR_DG_REACH) ; text = 'an axis has fewer cells than the reach of the wide operator'
   case(ELISHA_ERR_DG_AMR) ; text = 'the wide operator needs a uniform mesh and the spectral path'
   case(ELISHA_ERR_REDUNDANT) ; text = 'the redundant spectral solve is for the compact, non-isolated operator'
   case(ELISHA_ERR_GHOST_VALUES) ; text = 'boundary values need the hierarchical path and non-isolated boundaries'
   case(ELISHA_ERR_GHOST_COUNT) ; text = 'fewer boundary values than boundary faces'
   case(ELISHA_ERR_NOT_INITIALIZED) ; text = 'the solver is used before its setup'
   case(ELISHA_ERR_SHAPE) ; text = 'an array argument has the wrong shape or size'
   case(ELISHA_ERR_ENERGY_REDUNDANT) ; text = 'no energy with the redundant spectral solve'
   case(ELISHA_ERR_NOT_CONVERGED) ; text = 'an iterative solve reached max_iterations above its tolerance (phi holds' // &
      ' the last iterate, its ghosts filled)'
   case default ; text = 'an unknown status code'
   endselect
   endfunction elisha_status_text
endmodule elisha_status
