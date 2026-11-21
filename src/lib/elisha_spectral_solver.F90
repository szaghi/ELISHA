!< ELISHA, the spectral solver: the FFT Poisson solver on uniform block lattices.
module elisha_spectral_solver
!< ELISHA, the spectral solver: the FFT Poisson solver on uniform block lattices.
!<
!< **Provenance: ADAM's `adam_poisson_fft_object` (ADAM 669d0bdd), plumbing changed, arithmetic not.** What changed:
!< - ADAM's grid and field objects are ELISHA's mesh descriptor (`elisha_mesh_object`): `setup(mesh, comm, ...)`
!<   replaces `initialize(grid, field, ...)`, `solve_packed(mesh, buf, ...)` replaces `solve(field, q, iv, ...)` (which
!<   only packed one variable of a conservative array: the layout is the caller's) and `solve_packed(field, buf, ...)`;
!<   the levels and the cover of the domain are `mesh%validate`'s checks (`ELISHA_ERR_MESH_LEVELS`,
!<   `ELISHA_ERR_MESH_COVERAGE`), no longer the solver's own reductions;
!< - `MPI_COMM_WORLD` and ADAM's `mpih` are the caller's communication context, kept as the component `comm` (a copy:
!<   the caller owns the communicator); every collective goes through `elisha_comm_object`, so the serial build
!<   (`-DELISHA_NO_MPI`) has no MPI at all;
!< - `mpih%error_stop` is a status (`stat`, `errmsg`, `ELISHA_ERR_*`) returned up through every caller, and the
!<   argument refusals are `elisha_setup_checks`' pure checks;
!< - ADAM's private copies of the L0 math are ELISHA's, bitwise the same (`elisha_test_adam_parity_l0`,
!<   `elisha_test_lattice_green`): `elisha_axis_eigenvalue`, `elisha_dg_stencil`, `elisha_axis_transform_size` (with
!<   `elisha_smooth_size`), `elisha_axis_map`, `elisha_set_plane`, `elisha_lattice_green`.
!< What did not change: every floating-point expression, every loop and its order, every accumulation, the FFTW plans
!< and their flags, the layouts and the MPI calls (counts, displacements, datatypes, root 0, the `nr > 1` guards), so
!< the potential is meant to be ADAM's bitwise on every rank count (the tests hold it to the verbatim oracle,
!< `src/tests/adam_poisson_fft_object`). Issue numbers below (#81, #84, #87) are ADAM's.
!<
!< Solves `L_h phi = factor (f - mean(f))` for a cell field `f` on a mesh whose blocks all sit at one level and whose
!< active directions are all periodic. `L_h` is the second-order 7-point Laplacian (5-point in 2-D; a null direction
!< contributes nothing). The solve divides each Fourier mode by the eigenvalue of `L_h` itself,
!< `lambda_k = sum_d (2 cos(2 pi k_d / N_d) - 2) / dx_d^2`, so the discrete residual of the returned `phi` is at
!< round-off; the modes with `lambda_k = 0` (the mean, and every mode along a null direction) are set to zero.
!<
!< **Data path: the distributed slab transform (ADAM issue #84, E5d).** Rank r owns the z-planes [zr(r), zr(r+1)) of
!< the transform array (`m` cells per direction: the grid, its extensions, its padding). Every block plane goes to the
!< owners of the target planes that map to it, and each rank builds its planes by the per-axis maps (`tmap`, `tsgn`:
!< a target cell's source cell and sign, the images of `extend_global` and the zero padding). The planes are transformed
!< in 2-D (r2c), transposed to z-lines (`alltoallv_c`, the y-rows [yr(r), yr(r+1)) on rank r) and transformed in 1-D,
!< divided by the eigenvalues (or multiplied by the kernel's spectrum, kept in the pencil layout), and back. Every
!< block then fetches its potential with all its ghost layers from the owners of its planes (wrapped modulo `m`), so
!< no ghost exchange follows; the energy sums on the owners. One plan for one plane and one for one row of lines,
!< executed on every plane and row in turn through fixed buffers (`slab_planes`, `slab_lines`, the backend hooks):
!< the arithmetic of a value does not depend on how many planes or rows a rank holds, so the potential is bitwise the
!< same on every rank count. At most `m(3)` ranks hold planes (a pencil layout would lift it). PRISM's CG solves
!< (`pcg`, below), the multigrid's redundant root and the CG's sub-problems keep the rank-0 path of the first version
!< (ADAM issue #81, D-81-2): rank 0 gathers the source (`gatherv_r`), places it, transforms (`transform`) and scatters
!< every block's `phi` (`scatterv_r`); `global_potential` returns the transform array on rank 0 in either path
!< (diagnostics).
!<
!< **Backends.** This object transforms with FFTW 3 (`-D_FFTW`, `-lfftw3`; FFTW_ESTIMATE plans, deterministic). A
!< device backend (cuFFT, ELISHA M5) extends it and overrides the slab hooks (`slab_make_plans`, `slab_planes`,
!< `slab_lines`, `slab_destroy_plans`, ...) and the rank-0 ones (`make_plans`, `transform`, `destroy_plans`). Built
!< without an FFT library, `setup` returns `ELISHA_ERR_NO_FFT`.
!<
!< **Entry points.** `solve_packed` takes the interior source already packed (block-major, `i` fastest: what a device
!< kernel produces); `solve_blocks` takes an explicit list of blocks (the multigrid's root). Both can return
!< `energy = sum_cells f Phi` (summed on the slab owners, every rank gets it): `W = energy dV / 2` is the
!< gravitational energy of self-gravity on a uniform grid.
!<
!< **Isolated boundaries (ADAM issue #81, M8-P4), Hockney and Eastwood.** `setup(..., isolated=.true.)` solves
!< `L_h phi = factor f` with `phi -> 0` at infinity (no mean removed): the source sits in a zero-padded array of
!< `m >= 2 (n + ngc)` cells per direction, and the circular convolution with the lattice Green's function of `L_h`
!< (`elisha_lattice_green`, `h^2 G(n)`) equals the free-space one on every cell the scatter reads, the ghost
!< layers included, since `m` exceeds the largest offset, `n - 1 + ngc`, twice. `m` is the smallest even size at or
!< above `2 (n + ngc)` with no prime factor above 13 (#84 E5b): the transform cost no longer depends on the factors of
!< `n + ngc` (medooza, n = 128: 262 = 2 x 131 cells cost FFTW 3.3 times as long as 264; the kernel entries beyond
!< the offsets the read cells need meet only the zero padding). The kernel's spectrum (`khat`, real: the kernel is
!< even) is computed once. Inside the domain the discrete residual is that of the kernel, about 1e-11 of the source.
!<
!< **Homogeneous Dirichlet boundaries (ADAM issue #87, P1)** are for the multigrid's coarsest level only
!< (`initialize_blocks(..., dirichlet=.true.)`): `phi = 0` at the centres of the first ghost layer. The source is
!< extended odd about those centres to `m = 2 (n + 1)` cells per active direction, and the periodic transform of `m`
!< cells solves it: the odd extension is the type-I sine transform, exact for the 7-point operator with zero ghosts.
!<
!< **Boundary kinds per face (ADAM issue #84, E1a).** `bc(2,3)` (`elisha_parameters`) gives each face a homogeneous
!< kind: an axis is periodic, or each of its faces is Dirichlet (`phi = 0` at the face) or Neumann (zero normal
!< derivative at the face), possibly different at the two ends. The source is extended odd about a Dirichlet face and
!< even about a Neumann one, over 2n cells when the two faces agree and 4n when they differ (a quarter-wave
!< extension), and the periodic transform of the extension solves the 7-point operator with those boundary
!< conditions exactly; the scatter reads the ghost layers from the extension, so they carry the conditions. The modes
!< of zero eigenvalue are dropped as on a periodic grid: without a Dirichlet face the source mean is removed (the
!< operator is singular). The energy sums the interior only. Inhomogeneous values are the clients' work.
!<
!< **The wide operator D(G) (ADAM issue #84, E2a).** `dg_order = p` (2, 4, 6, 8; 0, the default, is the compact
!< 7-point Laplacian) solves PRISM's elliptic operator instead: the order-p centred first derivative `D1` composed with
!< itself per axis (the stencil reaches `p` cells, as PRISM's D and G read it), eigenvalue
!< `-sum_d s_p(theta_d)^2 / dx_d^2` with `s_p(theta) = sum_j 2 a_j sin(j theta)` (the `a_j` of `D1`). Every centred
!< `D1` annihilates the checkerboard (`s_p(pi) = 0`, set exactly: `sin(pi)` is 1e-16, not 0), so the modes with every
!< `theta_d` in {0, pi} are in the kernel and dropped, Dirichlet faces or not: G of them is zero, so the gradient a
!< client takes is unique. The odd and even extensions are reflections, which the wide stencil commutes with, so the
!< kinds per face hold as for the compact operator (the scatter fills `p` ghost layers from the extension); PRISM's
!< zero-ghost Dirichlet face is not a reflection (below). Measured on ADAM's prototype: a compact preconditioner needs
!< O(N) iterations for D(G) (the checkerboard branch), so the direct solve here is the method on uniform grids.
!<
!< **Zero-ghost faces (ADAM issue #84, E2b).** `ELISHA_BC_ZERO_GHOSTS` (both faces of an axis) is PRISM's elliptic
!< Dirichlet face: every ghost layer zero. With the compact operator it is the ghost-Dirichlet kind (the 7-point stencil
!< reads one layer). With the wide D(G) it is no reflection: the operator is the stencil truncated, symmetric and
!< nonsingular, and no transform diagonalises it. Rank 0 then solves it, on the gathered array, by conjugate gradients
!< preconditioned by the wide p = 2 operator with the same faces, inverted exactly: along a zero-ghost axis that
!< operator splits into the two parity sub-lattices, each the compact Laplacian at spacing 2h with zero at the ghost
!< centres (the ghost-Dirichlet kind), along the other axes it is the p = 2 operator with their own kinds; the 2^z
!< sub-problems (z zero-ghost axes) are FFT solves by sub-solvers of this solver's own type (`sub`). Spectrally
!< equivalent (kappa <= 2.8 at p = 4, 7.1 at p = 8): 1, 17, 24, 30 iterations at p = 2, 4, 6, 8, independent of the
!< grid. The ghost layers returned carry the faces' rules (zero, reflection, periodic wrap).
!<
!< **Free-space ghosts (ADAM issue #84, E3c).** `ELISHA_BC_FREE_GHOSTS` is PRISM's exact/open face: every ghost layer
!< holds the continuum free-space potential of the source, `g(x) = -sum_y factor f(y) dV / (4 pi |x - y|)`. The
!< operator is the zero-ghost one (the two kinds mix on an axis, and share the CG path and its preconditioner); the
!< ghost values move to the right-hand side, `D(G)_0 u = factor f - D(G) u_g` (u_g: g in the ghost slabs, zero
!< inside), and the returned ghost layers carry g. Rank 0 sums over the gathered source (`free_ghosts`), so the values
!< do not depend on the decomposition; the cost is O(ghost-slab cells x cells), a direct sum. The wide D(G) only.
!< Edges and corners of the ghost region stay zero: D(G) reads along the axis lines only.
!<
!< **Scope (refused with a status otherwise).** One level for every block, the blocks covering the domain. Without
!< `bc`, every active direction periodic. Isolated: 3-D, every direction non-periodic, cubic cells (the kernel is the
!< unit lattice's), the compact operator only. AMR is the hierarchical solver's, which calls this solver on its
!< coarsest level (`initialize_blocks`, `solve_blocks`).
use, intrinsic :: iso_c_binding
use penf, only : I4P, I8P, str
use elisha_kind_parameters, only : R_P
use elisha_comm, only : elisha_comm_object
use elisha_mesh, only : elisha_mesh_object
use elisha_parameters, only : ELISHA_BC_PERIODIC, ELISHA_BC_DIRICHLET, ELISHA_BC_NEUMANN, ELISHA_BC_DIRICHLET_GHOST, &
                              ELISHA_BC_ZERO_GHOSTS, ELISHA_BC_FREE_GHOSTS, ELISHA_PATH_SPECTRAL
use elisha_status, only : ELISHA_OK, ELISHA_ERR_NO_FFT, ELISHA_ERR_FFT_PLAN, ELISHA_ERR_MESH_LEVELS,           &
                          ELISHA_ERR_MESH_CHANGED, ELISHA_ERR_NOT_INITIALIZED, ELISHA_ERR_SHAPE,              &
                          ELISHA_ERR_ENERGY_REDUNDANT, ELISHA_ERR_ARGUMENT, ELISHA_ERR_NOT_CONVERGED,         &
                          elisha_set_error
use elisha_setup_checks, only : elisha_check_bc, elisha_check_dg_order, elisha_check_isolated, elisha_check_path, &
                                elisha_check_periodicity, elisha_check_reach, elisha_check_unbounded
use elisha_spectral, only : elisha_axis_eigenvalue, elisha_axis_transform_size, elisha_dg_stencil
use elisha_extension, only : elisha_axis_map, elisha_set_plane
use elisha_lattice_green_library, only : elisha_lattice_green
implicit none
private
public :: elisha_spectral_solver_object

type :: fft_slot
   !< One sub-solver of the E2b preconditioner (ADAM issue #84). A scalar polymorphic pointer in a slot, not a
   !< polymorphic array component: nvfortran 26.9 rejects a recursive allocatable component (NVFORTRAN-S-0155) and
   !< crashed deallocating a polymorphic pointer array allocated with `mold` (pgf90_dealloc_mbr03a, on ADAM's cuFFT
   !< type). Owned: `destroy` frees it.
   class(elisha_spectral_solver_object), pointer :: s=>null() !< The sub-solver.
endtype fft_slot

type :: elisha_spectral_solver_object
   !< FFT Poisson solver on uniform block lattices (see the module).
   logical                   :: is_initialized=.false. !< Initialization status.
   type(elisha_comm_object)  :: comm                   !< The caller's communication context (a copy: not owned).
   integer(I4P)              :: n(3)=1_I4P             !< Global cells per direction.
   integer(I4P)              :: m(3)=1_I4P             !< Transform cells per direction: n, or >= 2 (n + ngc) isolated.
   logical                   :: isolated=.false.       !< Isolated boundaries (zero padding), else periodic.
   logical                   :: dirichlet=.false.      !< Homogeneous Dirichlet boundaries (odd extension, #87).
   integer(I4P)              :: bc(2,3)=ELISHA_BC_PERIODIC !< Kinds per face [lo/hi, axis] (issue #84, E1a).
   logical                   :: extended=.false.       !< Some axis is extended (Dirichlet or Neumann faces).
   integer(I4P)              :: dg_order=0_I4P         !< 0: the compact Laplacian; p: the wide D(G) of order p (E2a).
   integer(I4P)              :: axis_order(3)=0_I4P    !< The operator per axis (dg_order, but in E2b's sub-problems).
   ! zero-ghost faces with the wide operator (issue #84, E2b): preconditioned CG on rank 0
   logical                   :: pcg=.false.            !< Some axis has zero-ghost faces and the operator is wide.
   logical                   :: free=.false.           !< Some face has free-space ghosts (E3c).
   real(R_P), allocatable    :: g_free(:,:,:)          !< Rank 0, E3c: the free-space ghost slabs, zero elsewhere.
   logical                   :: null_xyz(3)=.false.    !< Null directions.
   real(R_P)                 :: stencil(-8:8)=0._R_P   !< The 1-D D1 D1 stencil (times h^2), offsets -p..p.
   real(R_P)                 :: tolerance=1.e-12_R_P   !< Relative max-norm residual to stop.
   integer(I4P)              :: max_iterations=200_I4P !< CG iterations at most.
   logical                   :: converged=.true.       !< The last CG solve reached its tolerance (every rank); else
                                                       !< ELISHA_ERR_NOT_CONVERGED.
   integer(I4P)              :: iterations=0_I4P       !< Iterations of the last solve.
   real(R_P)                 :: residual=0._R_P        !< Relative residual of the last solve.
   logical                   :: redundant=.false.      !< Every rank holds the global arrays and solves (#84 E5c).
   real(R_P)                 :: wall(3)=0._R_P         !< Wall time (s) of `solve_blocks`, accumulated over calls on
                                                       !< this rank: gather, rank-0 solve, scatter (#84 E5; the
                                                       !< caller resets it).
   integer(I4P)              :: nsub=0_I4P             !< Preconditioner sub-problems, 2^z.
   integer(I4P),              allocatable :: sub_par(:,:) !< Parity per axis of each sub-problem [3,nsub] (-1: no split).
   type(fft_slot),            allocatable :: sub(:)       !< The preconditioner's sub-problems [nsub] (see fft_slot).
   integer(I4P)              :: nt(3)=1_I4P            !< Cells of the periodic transform the eigenvalues refer to: n,
                                                       !< or m with dirichlet.
   integer(I4P)              :: nijk(3)=1_I4P          !< Cells per block per direction.
   integer(I4P)              :: ngc=0_I4P              !< Ghost cells per side.
   integer(I4P)              :: level=0_I4P            !< Level of every block.
   real(R_P)                 :: dxyz(3)=0._R_P         !< Cell spacing per direction.
   real(R_P),    allocatable :: lam1(:)                !< Eigenvalue part along x [0:n(1)/2].
   real(R_P),    allocatable :: lam2(:)                !< Eigenvalue part along y [0:n(2)-1].
   real(R_P),    allocatable :: lam3(:)                !< Eigenvalue part along z [0:n(3)-1].
   real(R_P),    allocatable :: f_global(:,:,:)        !< Rank 0: the source, then the potential [m(1),m(2),m(3)].
   real(R_P),    allocatable :: f_source(:,:,:)        !< Rank 0: a copy of the source, for the energy.
   complex(R_P), allocatable :: f_hat(:,:,:)           !< Rank 0: the transform [m(1)/2+1,m(2),m(3)].
   real(R_P),    allocatable :: khat(:,:,:)            !< Rank 0, isolated: the kernel's spectrum over prod(m).
   type(c_ptr)               :: plan_r2c=c_null_ptr    !< FFTW forward plan (rank 0).
   type(c_ptr)               :: plan_c2r=c_null_ptr    !< FFTW backward plan (rank 0).
   ! the distributed slab transform (issue #84, E5d)
   logical                   :: slab=.false.           !< Every rank transforms its z-planes (see the module).
   integer(I4P)              :: h1=1_I4P               !< Complex cells along x after r2c, m(1)/2 + 1.
   integer(I4P), allocatable :: zr(:)                  !< First z-plane (0-based) of each rank's slab [0:ranks].
   integer(I4P), allocatable :: yr(:)                  !< First y-row (0-based) of each rank's pencils [0:ranks].
   integer(I4P), allocatable :: zown(:)                !< Owner rank of each z-plane [0:m(3)-1].
   integer(I4P), allocatable :: tmap(:,:)              !< Per axis, target cell -> source cell, -1 for zero [0:,3].
   real(R_P),    allocatable :: tsgn(:,:)              !< Per axis, the sign of the image [0:,3].
   real(R_P),    allocatable :: slab_r(:,:,:)          !< This rank's z-planes [m(1),m(2),planes].
   complex(R_P), allocatable :: slab_c(:,:,:)          !< Their 2-D transforms [h1,m(2),planes].
   complex(R_P), allocatable :: pen(:,:,:)             !< This rank's z-lines [m(3),h1,rows].
   real(R_P),    allocatable :: khat_pen(:,:,:)        !< Isolated: the kernel's spectrum in pencils [m(3),h1,rows].
   real(R_P),    allocatable :: w_r(:,:)               !< FFTW: the plane buffer [m(1),m(2)].
   complex(R_P), allocatable :: w_c(:,:)               !< FFTW: the transformed plane buffer [h1,m(2)].
   complex(R_P), allocatable :: w_l(:,:)               !< FFTW: the row-of-lines buffer [m(3),h1].
   type(c_ptr)               :: plan_pr2c=c_null_ptr   !< FFTW: one plane, r2c.
   type(c_ptr)               :: plan_pc2r=c_null_ptr   !< FFTW: one plane, c2r.
   type(c_ptr)               :: plan_lfwd=c_null_ptr   !< FFTW: one row of z-lines, forward.
   type(c_ptr)               :: plan_lbwd=c_null_ptr   !< FFTW: one row of z-lines, backward.
   contains
      ! public methods
      procedure, pass(self) :: destroy    !< Destroy the solver.
      procedure, pass(self) :: setup      !< Check the mesh and the arguments, size the solver, make the plans.
      procedure, pass(self) :: solve_packed !< Solve for phi from the packed interior source (host or device arrays).
      procedure, pass(self) :: initialize_blocks !< Size the solver from block counts and spacing (no mesh).
      procedure, pass(self) :: solve_blocks      !< Solve from an explicit list of blocks (coordinates and source).
      ! private methods
      procedure, pass(self), private :: make_kernel !< Rank 0, isolated: the kernel's spectrum.
      procedure, pass(self), private :: extend_global  !< Rank 0: the source extended over the transform cells.
      procedure, pass(self), private :: apply_wide     !< Rank 0: y = D(G) x with the faces' ghost rules (E2b).
      procedure, pass(self), private :: fill_ghosts    !< Rank 0: the ghost layers of a padded array by the faces.
      procedure, pass(self), private :: precondition   !< Rank 0: z = M^-1 r, the sub-lattice solves (E2b).
      procedure, pass(self), private :: solve_pcg      !< Rank 0: f_global <- factor D(G)^-1 f_global by PCG (E2b).
      procedure, pass(self), private :: free_ghosts    !< Rank 0: g_free, the free-space ghost slabs (E3c).
      procedure, pass(self), private :: subtract_free_ghosts !< Rank 0: f <- f - D(G) u_g (E3c).
      procedure, pass(self), private :: put_free_ghosts      !< Rank 0: the free-space ghost slabs into a padded array.
      procedure, pass(self), private :: solve_everywhere     !< `solve_blocks` with `redundant`: every rank solves.
      procedure, pass(self), private :: solve_packed_host    !< `solve_packed` on host arrays.
      procedure, pass(self), private :: slab_setup           !< The slab layout, maps, arrays and plans (E5d).
      procedure, pass(self), private :: solve_slabs          !< `solve_blocks` with `slab`: the distributed solve.
      procedure, pass(self), private :: slab_transform       !< slab_r <- factor * L^-1 slab_r, distributed.
      procedure, pass(self)          :: global_potential     !< Gather the slab solution to rank 0 (diagnostics).
      ! backend hooks
      procedure, pass(self) :: destroy_plans    !< Destroy the transform plans.
      procedure, pass(self) :: make_plans       !< Make the transform plans (rank 0).
      procedure, pass(self) :: transform        !< f_global <- factor * L_h^-1 f_global (rank 0).
      procedure, pass(self) :: transform_kernel !< khat <- the spectrum of f_global over prod(m) (rank 0, isolated).
      procedure, nopass     :: root_cells_max   !< Most cells of a multigrid root solved by this backend (#84 E5c, E6b).
      procedure, pass(self) :: slab_make_plans    !< Make the slab plans: one plane, one row of lines (E5d).
      procedure, pass(self) :: slab_destroy_plans !< Destroy the slab plans.
      procedure, pass(self) :: slab_planes        !< 2-D transform of every local plane, slab_r <-> slab_c.
      procedure, pass(self) :: slab_lines         !< 1-D transform of every local z-line, in place in pen.
      procedure, pass(self) :: slab_transpose     !< slab_c <-> pen (alltoallv_c).
      procedure, pass(self) :: slab_spectral      !< pen <- the spectral solve (eigenvalues or kernel) of pen.
      procedure, pass(self) :: slab_keep_kernel   !< khat_pen <- pen over prod(m) (isolated, at setup).
      procedure, pass(self) :: slab_update        !< slab_r between the host and where the stages run.
      procedure, pass(self) :: has_device_redundant   !< The redundant solve can run on device arrays (M7, C4).
      procedure, pass(self) :: solve_redundant_device !< `solve_everywhere` on device arrays (M7, C4).
      procedure, pass(self) :: stage_source    !< `solve_packed(on_device)`: the device source to a host copy (C4).
      procedure, pass(self) :: stage_potential !< `solve_packed(on_device)`: phi between the device and the host.
endtype elisha_spectral_solver_object

#ifdef _FFTW
integer(c_int), parameter :: FFTW_ESTIMATE=64_c_int !< FFTW planner flag, `fftw3.f03`.
integer(c_int), parameter :: FFTW_FORWARD=-1_c_int  !< FFTW sign, `fftw3.f03`.
integer(c_int), parameter :: FFTW_BACKWARD=1_c_int  !< FFTW sign, `fftw3.f03`.

interface
   type(c_ptr) function fftw_plan_dft_r2c_3d(n0, n1, n2, in, out, flags) bind(C, name='fftw_plan_dft_r2c_3d')
   !< FFTW: plan a 3-D real-to-complex transform (row-major sizes).
   import :: c_ptr, c_int, c_double, c_double_complex
   integer(c_int), value     :: n0, n1, n2 !< Sizes, slowest first.
   real(c_double)            :: in(*)      !< Input.
   complex(c_double_complex) :: out(*)     !< Output.
   integer(c_int), value     :: flags      !< Planner flags.
   endfunction fftw_plan_dft_r2c_3d

   type(c_ptr) function fftw_plan_dft_c2r_3d(n0, n1, n2, in, out, flags) bind(C, name='fftw_plan_dft_c2r_3d')
   !< FFTW: plan a 3-D complex-to-real transform (row-major sizes).
   import :: c_ptr, c_int, c_double, c_double_complex
   integer(c_int), value     :: n0, n1, n2 !< Sizes, slowest first.
   complex(c_double_complex) :: in(*)      !< Input.
   real(c_double)            :: out(*)     !< Output.
   integer(c_int), value     :: flags      !< Planner flags.
   endfunction fftw_plan_dft_c2r_3d

   subroutine fftw_execute_dft_r2c(p, in, out) bind(C, name='fftw_execute_dft_r2c')
   !< FFTW: execute a real-to-complex plan on the given arrays.
   import :: c_ptr, c_double, c_double_complex
   type(c_ptr), value        :: p      !< Plan.
   real(c_double)            :: in(*)  !< Input.
   complex(c_double_complex) :: out(*) !< Output.
   endsubroutine fftw_execute_dft_r2c

   subroutine fftw_execute_dft_c2r(p, in, out) bind(C, name='fftw_execute_dft_c2r')
   !< FFTW: execute a complex-to-real plan on the given arrays.
   import :: c_ptr, c_double, c_double_complex
   type(c_ptr), value        :: p      !< Plan.
   complex(c_double_complex) :: in(*)  !< Input.
   real(c_double)            :: out(*) !< Output.
   endsubroutine fftw_execute_dft_c2r

   type(c_ptr) function fftw_plan_dft_r2c_2d(n0, n1, in, out, flags) bind(C, name='fftw_plan_dft_r2c_2d')
   !< FFTW: plan a 2-D real-to-complex transform (row-major sizes).
   import :: c_ptr, c_int, c_double, c_double_complex
   integer(c_int), value     :: n0, n1 !< Sizes, slowest first.
   real(c_double)            :: in(*)  !< Input.
   complex(c_double_complex) :: out(*) !< Output.
   integer(c_int), value     :: flags  !< Planner flags.
   endfunction fftw_plan_dft_r2c_2d

   type(c_ptr) function fftw_plan_dft_c2r_2d(n0, n1, in, out, flags) bind(C, name='fftw_plan_dft_c2r_2d')
   !< FFTW: plan a 2-D complex-to-real transform (row-major sizes).
   import :: c_ptr, c_int, c_double, c_double_complex
   integer(c_int), value     :: n0, n1 !< Sizes, slowest first.
   complex(c_double_complex) :: in(*)  !< Input.
   real(c_double)            :: out(*) !< Output.
   integer(c_int), value     :: flags  !< Planner flags.
   endfunction fftw_plan_dft_c2r_2d

   type(c_ptr) function fftw_plan_many_dft(rnk, n, howmany, in, inembed, istride, idist, out, onembed, ostride, &
                                           odist, sgn, flags) bind(C, name='fftw_plan_many_dft')
   !< FFTW: plan a batch of complex transforms.
   import :: c_ptr, c_int, c_double_complex
   integer(c_int), value     :: rnk             !< Rank of each transform.
   integer(c_int)            :: n(*)            !< Sizes.
   integer(c_int), value     :: howmany         !< Transforms.
   complex(c_double_complex) :: in(*)           !< Input.
   type(c_ptr), value        :: inembed         !< NULL: n.
   integer(c_int), value     :: istride, idist  !< Input stride and distance.
   complex(c_double_complex) :: out(*)          !< Output.
   type(c_ptr), value        :: onembed         !< NULL: n.
   integer(c_int), value     :: ostride, odist  !< Output stride and distance.
   integer(c_int), value     :: sgn             !< FFTW_FORWARD or FFTW_BACKWARD.
   integer(c_int), value     :: flags           !< Planner flags.
   endfunction fftw_plan_many_dft

   subroutine fftw_execute_dft(p, in, out) bind(C, name='fftw_execute_dft')
   !< FFTW: execute a complex plan on the given arrays.
   import :: c_ptr, c_double_complex
   type(c_ptr), value        :: p      !< Plan.
   complex(c_double_complex) :: in(*)  !< Input.
   complex(c_double_complex) :: out(*) !< Output.
   endsubroutine fftw_execute_dft

   subroutine fftw_destroy_plan(p) bind(C, name='fftw_destroy_plan')
   !< FFTW: destroy a plan.
   import :: c_ptr
   type(c_ptr), value :: p !< Plan.
   endsubroutine fftw_destroy_plan
endinterface
#endif

contains
   ! public methods
   recursive subroutine destroy(self)
   !< Destroy the solver (and its sub-solvers).
   class(elisha_spectral_solver_object), intent(inout) :: self !< The solver.
   integer(I4P)                             :: i    !< Counter.

   if (self%is_initialized) call self%destroy_plans
   if (self%is_initialized .and. self%slab) call self%slab_destroy_plans
   if (allocated(self%zr))       deallocate(self%zr)
   if (allocated(self%yr))       deallocate(self%yr)
   if (allocated(self%zown))     deallocate(self%zown)
   if (allocated(self%tmap))     deallocate(self%tmap)
   if (allocated(self%tsgn))     deallocate(self%tsgn)
   if (allocated(self%slab_r))   deallocate(self%slab_r)
   if (allocated(self%slab_c))   deallocate(self%slab_c)
   if (allocated(self%pen))      deallocate(self%pen)
   if (allocated(self%khat_pen)) deallocate(self%khat_pen)
   if (allocated(self%w_r))      deallocate(self%w_r)
   if (allocated(self%w_c))      deallocate(self%w_c)
   if (allocated(self%w_l))      deallocate(self%w_l)
   self%slab = .false. ; self%h1 = 1_I4P
   if (allocated(self%lam1))     deallocate(self%lam1)
   if (allocated(self%lam2))     deallocate(self%lam2)
   if (allocated(self%lam3))     deallocate(self%lam3)
   if (allocated(self%f_global)) deallocate(self%f_global)
   if (allocated(self%f_source)) deallocate(self%f_source)
   if (allocated(self%f_hat))    deallocate(self%f_hat)
   if (allocated(self%khat))     deallocate(self%khat)
   if (allocated(self%sub)) then
      do i=1, size(self%sub)
         if (associated(self%sub(i)%s)) then
            call self%sub(i)%s%destroy
            deallocate(self%sub(i)%s)
         endif
      enddo
      deallocate(self%sub)
   endif
   if (allocated(self%sub_par))  deallocate(self%sub_par)
   if (allocated(self%g_free))   deallocate(self%g_free)
   self%free = .false.
   self%pcg = .false. ; self%nsub = 0_I4P ; self%iterations = 0_I4P ; self%residual = 0._R_P
   self%converged = .true.
   self%dg_order = 0_I4P ; self%axis_order = 0_I4P ; self%stencil = 0._R_P ; self%null_xyz = .false.
   self%n = 1_I4P ; self%m = 1_I4P ; self%nijk = 1_I4P ; self%ngc = 0_I4P ; self%level = 0_I4P ; self%dxyz = 0._R_P
   self%isolated = .false.
   self%redundant = .false.
   self%dirichlet = .false.
   self%bc = ELISHA_BC_PERIODIC
   self%extended = .false.
   self%nt = 1_I4P
   self%comm = elisha_comm_object() ! a copy: the caller frees its communicator
   self%is_initialized = .false.
   endsubroutine destroy

   subroutine setup(self, mesh, comm, stat, errmsg, isolated, bc, dg_order)
   !< Check the mesh (one level, the domain covered) and the boundaries, size the solver, make the plans and, isolated,
   !< the kernel. Collective over `comm`, which the solver keeps (a copy: the caller frees the communicator after
   !< `destroy`). The mesh is validated here with `uniform` unless the caller validated it already (then its level range
   !< is checked: one level, `ELISHA_ERR_MESH_LEVELS`); `validate` replaces ADAM's level and cover reductions. `bc`
   !< (ADAM issue #84, E1a): the kind of every face; a periodic axis of the mesh must be periodic in `bc` and the other
   !< way round. Without it every active axis must be periodic (or the solve isolated, which takes no `bc`).
   class(elisha_spectral_solver_object), intent(inout)         :: self     !< The solver.
   type(elisha_mesh_object),             intent(inout)         :: mesh     !< The mesh (validated here if not yet).
   type(elisha_comm_object),             intent(in)            :: comm     !< Communication context (initialised).
   integer(I4P),                         intent(out)           :: stat     !< Status.
   character(:), allocatable,            intent(out), optional :: errmsg   !< Message.
   logical,      optional,               intent(in)            :: isolated !< Isolated boundaries (default: periodic).
   integer(I4P), optional,               intent(in)            :: bc(2,3)  !< Kinds per face [lo/hi, axis] (default
                                                                           !< periodic).
   integer(I4P), optional,               intent(in)            :: dg_order !< 0 compact (default); 2, 4, 6, 8: the wide
                                                                           !< D(G) (E2a).
   integer(I4P)                                                :: nblk(3)  !< Blocks per direction at the level.
   integer(I4P)                                                :: dg_order_ !< Copy of the argument.
   integer(I4P)                                                :: d        !< Counter.
   character(64)                                               :: w        !< Message buffer.

   call self%destroy
   if (.not.comm%is_initialized) then
      call elisha_set_error(stat, errmsg, ELISHA_ERR_NOT_INITIALIZED, &
                            'elisha_spectral_solver_object%setup: the communication context is not initialised')
      return
   endif
   if (.not.mesh%is_validated) then
      call mesh%validate(comm, stat, errmsg, uniform=.true.)
      if (stat /= ELISHA_OK) return
   elseif (mesh%lmin /= mesh%lmax) then
      write(w, '(A,I0,A,I0)') 'the blocks sit at levels ', mesh%lmin, ' to ', mesh%lmax
      call elisha_set_error(stat, errmsg, ELISHA_ERR_MESH_LEVELS, 'elisha_spectral_solver_object%setup: '//trim(w)// &
                            '; the spectral solver needs a uniform mesh (one level)')
      return
   endif
   self%isolated = .false. ; if (present(isolated)) self%isolated = isolated
   self%nijk = mesh%nijk
   self%ngc  = mesh%ngc
   self%level = mesh%lmin
   self%dxyz  = mesh%cell_dx(self%level)
   do d=1, 3
      nblk(d) = int(mesh%blocks_along(self%level, d), I4P)
   enddo
   dg_order_ = 0_I4P ; if (present(dg_order)) dg_order_ = dg_order
   call elisha_check_dg_order(dg_order_, stat, errmsg) ; if (stat /= ELISHA_OK) return
   call elisha_check_isolated(self%isolated, present(bc), dg_order_, mesh%null, mesh%periodic, self%dxyz, stat, errmsg)
   if (stat /= ELISHA_OK) return
   if (.not.self%isolated) then
      if (present(bc)) then
         call elisha_check_periodicity(bc, mesh%periodic, mesh%null, stat, errmsg)
      else
         call elisha_check_unbounded(ELISHA_PATH_SPECTRAL, mesh%periodic, mesh%null, stat, errmsg)
      endif
      if (stat /= ELISHA_OK) return
   endif
   if (present(bc)) then ! not isolated (refused above)
      call self%initialize_blocks(nblk=nblk, nijk=self%nijk, ngc=self%ngc, dxyz=self%dxyz, level=self%level,         &
                                  null_xyz=mesh%null, isolated=self%isolated, comm=comm, stat=stat, errmsg=errmsg, &
                                  bc=bc, dg_order=dg_order)
   else
      call self%initialize_blocks(nblk=nblk, nijk=self%nijk, ngc=self%ngc, dxyz=self%dxyz, level=self%level,         &
                                  null_xyz=mesh%null, isolated=self%isolated, comm=comm, stat=stat, errmsg=errmsg, &
                                  dg_order=dg_order)
   endif
   endsubroutine setup

   recursive subroutine initialize_blocks(self, nblk, nijk, ngc, dxyz, level, null_xyz, isolated, comm, stat, errmsg, &
                                          dirichlet, bc, dg_order, axis_dg_order, redundant)
   !< Size the solver for a uniform grid of `nblk` blocks per direction, `nijk` cells each, at `level` with spacing
   !< `dxyz`, and make the plans (and, isolated, the kernel) on rank 0. No check of a mesh: the caller owns the scope
   !< (the multigrid's coarsest level calls it directly, ADAM issue #81 P5; with `dirichlet`, issue #87). Collective
   !< over `comm`, which the solver keeps (a copy); every rank returns the same status.
   !<
   !< `bc` (ADAM issue #84, E1a) gives the boundary kind of every face (`elisha_parameters`), homogeneous: an axis is
   !< periodic (both faces), or its faces are Dirichlet/Neumann at the face (the source extended odd/even about each
   !< face: 2n cells when the two faces agree, 4n when they differ), or ghost-Dirichlet (both faces, odd about the ghost
   !< centres, 2(n+1) cells: what `dirichlet` sets on every active axis). Periodic against non-periodic, and
   !< ghost-Dirichlet against a face kind, on one axis are refused (`elisha_check_bc`); a null axis takes no extension.
   !<
   !< `redundant` (ADAM issue #84, E5c): every rank holds the global arrays and the plans, and `solve_blocks` gathers
   !< the source to every rank (`allgatherv_layout_r`), transforms on every rank and keeps each rank's own blocks: no
   !< scatter, and no rank waits on another's transform. For a small grid (the multigrid's root); the compact operator
   !< only, not isolated, no energy.
   class(elisha_spectral_solver_object), intent(inout)         :: self        !< The solver.
   integer(I4P),                         intent(in)            :: nblk(3)     !< Blocks per direction.
   integer(I4P),                         intent(in)            :: nijk(3)     !< Cells per block per direction.
   integer(I4P),                         intent(in)            :: ngc         !< Ghost cells per side (isolated: the
                                                                              !< padding margin).
   real(R_P),                            intent(in)            :: dxyz(3)     !< Cell spacing.
   integer(I4P),                         intent(in)            :: level       !< Level of every block.
   logical,                              intent(in)            :: null_xyz(3) !< Null directions.
   logical,                              intent(in)            :: isolated    !< Isolated boundaries, else periodic.
   type(elisha_comm_object),             intent(in)            :: comm        !< Communication context (initialised).
   integer(I4P),                         intent(out)           :: stat        !< Status.
   character(:), allocatable,            intent(out), optional :: errmsg      !< Message.
   logical,                              intent(in),  optional :: dirichlet   !< Homogeneous ghost-Dirichlet everywhere
                                                                              !< (default false).
   integer(I4P),                         intent(in),  optional :: bc(2,3)     !< Kinds per face [lo/hi, axis] (default
                                                                              !< periodic).
   integer(I4P),                         intent(in),  optional :: dg_order    !< 0 compact (default); 2, 4, 6, 8: D(G).
   integer(I4P),                         intent(in),  optional :: axis_dg_order(3) !< The operator per axis (E2b's
                                                                                   !< sub-problems).
   logical,                              intent(in),  optional :: redundant   !< Every rank solves (default false).
   integer(I4P)                                    :: nblk_(3), nijk_(3), ngc_, level_ !< Copies of the arguments.
   integer(I4P)                                    :: dg_order_   !< Copy of the argument.
   integer(I4P)                                    :: axis_order_(3) !< Copy of the argument.
   real(R_P)                                       :: dxyz_(3)    !< Copy of the argument.
   logical                                         :: no_periodic(3) !< No periodic axis (elisha_check_isolated).
   logical                                         :: null_(3), isolated_, dirichlet_ !< Copies of the arguments.
   integer(I4P)                                    :: bc_(2,3)    !< Copy of the argument.
   type(elisha_comm_object)                        :: comm_       !< Copy of the argument.
   integer(I4P)                                    :: k, d        !< Counters.

   ! copy first: the arguments may be components of self (setup passes them), which destroy resets
   nblk_ = nblk ; nijk_ = nijk ; ngc_ = ngc ; level_ = level ; dxyz_ = dxyz ; null_ = null_xyz ; isolated_ = isolated
   comm_ = comm
   if (.not.comm_%is_initialized) then
      call elisha_set_error(stat, errmsg, ELISHA_ERR_NOT_INITIALIZED, 'elisha_spectral_solver_object%initialize_blocks: '// &
                            'the communication context is not initialised')
      return
   endif
   dirichlet_ = .false.
   if (present(dirichlet)) dirichlet_ = dirichlet
   bc_ = ELISHA_BC_PERIODIC
   if (present(bc)) bc_ = bc
   if (dirichlet_) bc_ = ELISHA_BC_DIRICHLET_GHOST
   dg_order_ = 0_I4P
   if (present(dg_order)) dg_order_ = dg_order
   call elisha_check_dg_order(dg_order_, stat, errmsg) ; if (stat /= ELISHA_OK) return
   no_periodic = .false. ! a named array, not a constructor: no argument temporary (ifx -check arg_temp_created)
   call elisha_check_isolated(isolated_, dirichlet_ .or. any(bc_ /= ELISHA_BC_PERIODIC), dg_order_, null_, &
                              no_periodic, dxyz_, stat, errmsg)
   if (stat /= ELISHA_OK) return
   axis_order_ = dg_order_
   if (present(axis_dg_order)) axis_order_ = axis_dg_order
   call elisha_check_bc(bc_, null_, axis_order_, stat, errmsg) ; if (stat /= ELISHA_OK) return
   ! zero ghosts: with the compact operator, the ghost-Dirichlet kind (one ghost layer read); wide, preconditioned CG
   if (all(axis_order_ == 0_I4P)) where (bc_ == ELISHA_BC_ZERO_GHOSTS) bc_ = ELISHA_BC_DIRICHLET_GHOST
   call self%destroy
   self%comm = comm_
   self%nijk = nijk_
   self%ngc = ngc_
   self%dxyz = dxyz_
   self%level = level_
   self%isolated = isolated_
   self%dirichlet = dirichlet_
   self%bc = bc_
   self%dg_order = dg_order_
   self%axis_order = axis_order_
   self%null_xyz = null_
   self%pcg = .false.
   do d=1, 3 ! a null axis has no ghosts the operator reads
      if (.not.null_(d)) self%pcg = self%pcg .or. any(self%bc(:,d) == ELISHA_BC_ZERO_GHOSTS .or. &
                                                    self%bc(:,d) == ELISHA_BC_FREE_GHOSTS)
   enddo
   self%free = any(self%bc == ELISHA_BC_FREE_GHOSTS)
   if (self%pcg) self%stencil = elisha_dg_stencil(self%dg_order)
   self%n = nblk_ * self%nijk
   call elisha_check_reach(self%n, null_, self%dg_order, self%pcg, stat, errmsg)
   if (stat /= ELISHA_OK) then
      call abandon
      return
   endif
   ! the transform cells per axis: n (periodic, null, the CG path), the extension of the face kinds (2 (n + 1)
   ! ghost-Dirichlet, 2n or 4n Dirichlet/Neumann), isolated at least 2 (n + ngc), the next 13-smooth size (#84 E5b:
   ! FFTW slows on large prime factors)
   do d=1, 3
      self%m(d) = elisha_axis_transform_size(self%n(d), self%ngc, self%bc(:,d), self%isolated, self%pcg, null_(d))
   enddo
   self%extended = (.not.self%isolated) .and. any(self%m /= self%n) ! never with pcg (m = n)
   self%nt = merge(self%n, self%m, self%isolated)
   ! eigenvalue parts of L_h per direction on the nt-cell periodic transform; zero along a null direction
   allocate(self%lam1(0:self%nt(1)/2), self%lam2(0:self%nt(2)-1), self%lam3(0:self%nt(3)-1))
   self%lam1 = 0._R_P ; self%lam2 = 0._R_P ; self%lam3 = 0._R_P
   if (.not.null_(1)) then
      do k=0, self%nt(1)/2
         self%lam1(k) = elisha_axis_eigenvalue(k, self%nt(1), self%dxyz(1), self%axis_order(1))
      enddo
   endif
   if (.not.null_(2)) then
      do k=0, self%nt(2)-1
         self%lam2(k) = elisha_axis_eigenvalue(k, self%nt(2), self%dxyz(2), self%axis_order(2))
      enddo
   endif
   if (.not.null_(3)) then
      do k=0, self%nt(3)-1
         self%lam3(k) = elisha_axis_eigenvalue(k, self%nt(3), self%dxyz(3), self%axis_order(3))
      enddo
   endif
   if (present(redundant)) self%redundant = redundant
   call elisha_check_path(ELISHA_PATH_SPECTRAL, .true., self%dg_order, self%bc, self%isolated, self%redundant, &
                          self%pcg, .false., stat, errmsg)
   if (stat /= ELISHA_OK) then
      call abandon
      return
   endif
   ! the distributed slab transform (#84 E5d), for every solve but PRISM's CG ones (rank 0, #84 decision 3), the
   ! redundant root and the CG's own sub-problems
   self%slab = .not.(self%pcg .or. self%redundant .or. present(axis_dg_order))
   if (self%slab) then
      call self%slab_setup(stat, errmsg) ! agrees on the status before its collectives
   elseif (self%comm%rank == 0 .or. self%redundant) then
      allocate(self%f_global(self%m(1),self%m(2),self%m(3)))
      allocate(self%f_source(self%m(1),self%m(2),self%m(3)))
      allocate(self%f_hat(self%m(1)/2+1,self%m(2),self%m(3)))
      self%f_global = 0._R_P
      self%f_source = 0._R_P
      self%f_hat = (0._R_P, 0._R_P)
      stat = ELISHA_OK
      if (.not.self%pcg) call self%make_plans(stat, errmsg) ! the CG solve transforms its sub-problems only
      if (self%isolated .and. stat == ELISHA_OK) call self%make_kernel(stat, errmsg)
   else
      stat = ELISHA_OK
   endif
   ! the plans fail on some ranks only (rank 0's path): every rank takes the worst status before going on
   call agree_status(self%comm, stat, errmsg, 'elisha_spectral_solver_object%initialize_blocks')
   if (stat == ELISHA_OK .and. self%pcg) call make_preconditioner
   if (stat /= ELISHA_OK) then
      call abandon
      return
   endif
   self%is_initialized = .true.
   contains
      subroutine abandon
      !< Free what was built on a failure (destroy frees the plans of an initialised solver only).
      self%is_initialized = .true.
      call self%destroy
      endsubroutine abandon

      subroutine make_preconditioner
      !< The 2^z sub-problems of the wide p = 2 operator (z zero-ghost axes): along a zero-ghost axis one parity class
      !< (ceil or floor of n/2 cells), the compact operator at spacing 2h with zero at the ghost centres; along the other
      !< axes the p = 2 operator with their kinds. Sub-solvers of this solver's own type (a device backend's too), one
      !< block each; their statuses are agreed over the ranks.
      integer(I4P) :: z, c, d_, bit       !< Zero-ghost axes, counters.
      integer(I4P) :: n_sub(3), ord_sub(3), bc_sub(2,3) !< A sub-problem's cells, operators, kinds.
      real(R_P)    :: dx_sub(3)           !< A sub-problem's spacing.
      integer(I4P) :: one_block(3)        !< One block per axis (a named array: no argument temporary).

      one_block = 1_I4P
      z = count([(zero_class(d_), d_=1, 3)])
      self%nsub = 2_I4P**z
      allocate(self%sub_par(3,self%nsub))
      allocate(self%sub(self%nsub))
      do c=1, self%nsub
         bit = 0
         do d_=1, 3
            self%sub_par(d_,c) = -1
            n_sub(d_) = self%n(d_) ; dx_sub(d_) = self%dxyz(d_) ; ord_sub(d_) = 2_I4P ; bc_sub(:,d_) = self%bc(:,d_)
            if (zero_class(d_)) then
               self%sub_par(d_,c) = ibits(c-1, bit, 1)
               bit = bit + 1
               n_sub(d_) = (self%n(d_) - self%sub_par(d_,c) + 1) / 2 ! cells i (1-based) with mod(i-1, 2) = parity
               dx_sub(d_) = 2._R_P * self%dxyz(d_) ; ord_sub(d_) = 0_I4P ; bc_sub(:,d_) = ELISHA_BC_DIRICHLET_GHOST
            endif
            if (null_(d_)) ord_sub(d_) = 0_I4P
         enddo
         allocate(self%sub(c)%s, mold=self)
         call self%sub(c)%s%initialize_blocks(nblk=one_block, nijk=n_sub, ngc=0_I4P, dxyz=dx_sub,     &
                                            level=self%level, null_xyz=null_, isolated=.false., comm=self%comm, &
                                            stat=stat, errmsg=errmsg, bc=bc_sub, dg_order=2_I4P,              &
                                            axis_dg_order=ord_sub)
         if (stat /= ELISHA_OK) return ! every rank alike (the sub-solver agreed)
      enddo
      endsubroutine make_preconditioner

      pure function zero_class(d_)
      !< The axis has zero-ghost (or free-space ghost) faces and is not null: its parity classes decouple.
      integer(I4P), intent(in) :: d_         !< Axis.
      logical                  :: zero_class !< Result.

      zero_class = .not.self%null_xyz(d_) .and. any(self%bc(:,d_) == ELISHA_BC_ZERO_GHOSTS .or. &
                                                    self%bc(:,d_) == ELISHA_BC_FREE_GHOSTS)
      endfunction zero_class
   endsubroutine initialize_blocks

   subroutine solve_packed(self, mesh, buf, factor, phi, stat, errmsg, energy, on_device)
   !< Solve from the packed interior source (see `solve_packed_host`). `on_device` (M7, audit C4): `buf` and `phi` are
   !< present on the device (OpenACC), a device backend's arrays: the source is staged to a host copy
   !< (`stage_source`), `phi` to the host and back (`stage_potential`; contiguous), around the host solve, so the result
   !< is bitwise the one from host arrays. This host solver has no device arrays: `on_device=.true.` returns
   !< `ELISHA_ERR_ARGUMENT` (an argument every rank passes alike, refused before any collective). The CG of the wide
   !< D(G) (zero- or free-ghost faces) that reaches `max_iterations` above its tolerance returns
   !< `ELISHA_ERR_NOT_CONVERGED` on every rank (`converged` false), `phi` the last iterate with its ghosts filled; ADAM
   !< printed a warning on rank 0 and returned as if converged.
   class(elisha_spectral_solver_object), intent(inout)         :: self   !< The solver.
   type(elisha_mesh_object),             intent(in)            :: mesh   !< The mesh (this rank's block coordinates).
   real(R_P),                            intent(in)            :: buf(1:) !< Packed interior source of this rank.
   real(R_P),                            intent(in)            :: factor !< Source factor (4 pi G for gravity).
   real(R_P),    intent(inout) :: phi(1-self%ngc:,1-self%ngc:,1-self%ngc:,1:) !< Potential [i,j,k,nb].
   integer(I4P),                         intent(out)           :: stat   !< Status.
   character(:), allocatable,            intent(out), optional :: errmsg !< Message.
   real(R_P),                            intent(out), optional :: energy !< sum_cells f phi (all ranks).
   logical,                              intent(in),  optional :: on_device !< buf and phi on the device (default: host).
   real(R_P), allocatable :: bufh(:) !< The host copy of a device source.
   logical                :: dev     !< on_device.
   integer(I4P)           :: code    !< Status of the staging.

   dev = .false. ; if (present(on_device)) dev = on_device
   self%converged = .true.
   if (.not.dev) then
      call self%solve_packed_host(mesh=mesh, buf=buf, factor=factor, phi=phi, stat=stat, errmsg=errmsg, energy=energy)
   else
      call self%stage_source(buf, bufh, code)
      if (code /= ELISHA_OK) then
         call elisha_set_error(stat, errmsg, code, 'elisha_spectral_solver_object%solve_packed: on_device needs a '// &
                               'device backend (the host solver has no device arrays)')
         return
      endif
      call self%stage_potential(phi, to_host=.true.)
      call self%solve_packed_host(mesh=mesh, buf=bufh, factor=factor, phi=phi, stat=stat, errmsg=errmsg, energy=energy)
      ! not converged (ELISHA notes E1c, decision C): phi is the last iterate, staged back as a converged one
      if (stat == ELISHA_OK .or. stat == ELISHA_ERR_NOT_CONVERGED) call self%stage_potential(phi, to_host=.false.)
   endif
   endsubroutine solve_packed

   subroutine solve_packed_host(self, mesh, buf, factor, phi, stat, errmsg, energy)
   !< Solve `L_h phi = factor (f - mean(f))` (periodic) or `L_h phi = factor f` (isolated) from the packed interior
   !< source of this rank's blocks (block-major, `i` fastest, `mesh%nb * ni * nj * nk` values); every ghost layer of phi
   !< is filled from the global solution (the periodic wrap, or the padded free-space solution). Collective: the
   !< argument checks (setup done, phi's shape, every block at the setup's level) are agreed over the ranks before any
   !< transfer.
   class(elisha_spectral_solver_object), intent(inout)         :: self   !< The solver.
   type(elisha_mesh_object),             intent(in)            :: mesh   !< The mesh (this rank's block coordinates).
   real(R_P),                            intent(in)            :: buf(1:) !< Packed interior source of this rank.
   real(R_P),                            intent(in)            :: factor !< Source factor (4 pi G for gravity).
   real(R_P),    intent(inout) :: phi(1-self%ngc:,1-self%ngc:,1-self%ngc:,1:) !< Potential [i,j,k,nb].
   integer(I4P),                         intent(out)           :: stat   !< Status.
   character(:), allocatable,            intent(out), optional :: errmsg !< Message.
   real(R_P),                            intent(out), optional :: energy !< sum_cells f phi (all ranks).
   integer(I4P)                                     :: nbl         !< Blocks of this rank.
   integer(I4P), allocatable                        :: coords(:,:) !< Block coordinates of this rank [3,nbl].
   real(R_P),    allocatable                        :: rbuf(:)     !< This rank's potential, ghosts included.
   integer(I4P)                                     :: ub(3)       !< Upper bounds of phi per direction.
   integer(I4P)                                     :: b, i, j, k, c !< Counters.

   if (.not.self%is_initialized) then ! every rank alike: setup is collective
      call elisha_set_error(stat, errmsg, ELISHA_ERR_NOT_INITIALIZED, &
                            'elisha_spectral_solver_object%solve_packed: called before setup')
      return
   endif
   stat = ELISHA_OK
   ub = self%nijk + self%ngc
   if (ubound(phi, dim=1) /= ub(1) .or. ubound(phi, dim=2) /= ub(2) .or. ubound(phi, dim=3) /= ub(3) .or. &
       size(phi, dim=4) < mesh%nb) &
      call elisha_set_error(stat, errmsg, ELISHA_ERR_SHAPE, 'elisha_spectral_solver_object%solve_packed: phi is not '// &
                            'shaped [1-ngc:n+ngc per direction, blocks]')
   nbl = mesh%nb
   allocate(coords(3,max(nbl,1)))
   do b=1, nbl
      if (stat /= ELISHA_OK) exit
      if (mesh%coords(4,b) /= self%level) call elisha_set_error(stat, errmsg, ELISHA_ERR_MESH_CHANGED, &
         'elisha_spectral_solver_object%solve_packed: a block left level '//trim(str(self%level, .true.))// &
         ' after setup (regrids need a new setup)')
      coords(:,b) = mesh%coords(1:3,b)
   enddo
   call agree_status(self%comm, stat, errmsg, 'elisha_spectral_solver_object%solve_packed')
   if (stat /= ELISHA_OK) return
   allocate(rbuf(max(nbl*product(self%nijk + 2 * self%ngc),1)))
   call self%solve_blocks(nbl=nbl, coords=coords, buf=buf, factor=factor, gw=self%ngc, out=rbuf, stat=stat, &
                          errmsg=errmsg, energy=energy)
   if (stat /= ELISHA_OK .and. stat /= ELISHA_ERR_NOT_CONVERGED) return ! not converged: the last iterate, unpacked
   c = 0
   do b=1, nbl
      do k=1-self%ngc, ub(3)
         do j=1-self%ngc, ub(2)
            do i=1-self%ngc, ub(1)
               c = c + 1
               phi(i,j,k,b) = rbuf(c)
            enddo
         enddo
      enddo
   enddo
   endsubroutine solve_packed_host

   subroutine solve_blocks(self, nbl, coords, buf, factor, gw, out, stat, errmsg, energy)
   !< The solve on an explicit list of this rank's blocks: their integer coordinates at the solver's level and their
   !< packed interior source (block-major, `i` fastest); `out` returns each block's potential with `gw` ghost layers
   !< (block-major, `i` fastest, from `1-gw` to `n+gw`), filled from the global solution. Optionally the energy
   !< `sum_cells f phi` (rank 0 computes it, every rank gets it). Collective; the caller (the multigrid's root, or
   !< `solve_packed`) owns the layout: only the rank-0 path checks the sizes of `buf` and `out`, as ADAM did, and agrees
   !< on that status over the ranks.
   class(elisha_spectral_solver_object), intent(inout)         :: self            !< The solver.
   integer(I4P),              intent(in)            :: nbl             !< Blocks of this rank.
   integer(I4P),              intent(in)            :: coords(1:,1:)   !< Block coordinates [3,nbl].
   real(R_P),                 intent(in)            :: buf(1:)         !< Packed interior source of this rank.
   real(R_P),                 intent(in)            :: factor          !< Source factor (4 pi G for gravity).
   integer(I4P),              intent(in)            :: gw              !< Ghost layers of `out`.
   real(R_P),                 intent(inout)         :: out(1:)         !< Packed potential, ghosts included.
   integer(I4P),              intent(out)           :: stat            !< Status.
   character(:), allocatable, intent(out), optional :: errmsg          !< Message.
   real(R_P),                 intent(out), optional :: energy          !< sum_cells f phi (all ranks).
   integer(I4P)                                     :: ncell           !< Interior cells per block.
   integer(I4P)                                     :: nghost          !< Cells per block, ghosts included.
   integer(I4P)                                     :: nr              !< Ranks.
   integer(I4P)                                     :: blocks_total    !< Rank 0: blocks over the ranks.
   integer(I4P), allocatable                        :: counts(:)       !< Rank 0: blocks per rank.
   integer(I4P), allocatable                        :: displs(:)       !< Rank 0: displacement per rank (blocks).
   integer(I4P), allocatable                        :: rcounts(:)      !< Rank 0: counts per rank (values).
   integer(I4P), allocatable                        :: rdispls(:)      !< Rank 0: displacements per rank (values).
   integer(I4P), allocatable                        :: coords_all(:,:) !< Rank 0: block coordinates of every rank.
   real(R_P),    allocatable                        :: buf_all(:)      !< Rank 0: buffer of every rank.
   real(R_P),    allocatable                        :: u_pad(:,:,:)    !< Rank 0, E2b: the solution with its ghosts.
   real(R_P)                                        :: e               !< sum_cells f phi.
   real(R_P)                                        :: e_(1)           !< e, as the broadcast's buffer.
   integer(I4P)                                     :: nc_(1)          !< 1: rank 0's CG did not converge.
   integer(I4P)                                     :: ub(3)           !< Upper bounds per direction.
   integer(I4P)                                     :: g(3)            !< Global cell index, wrapped.
   integer(I4P)                                     :: b, i, j, k, c, r !< Counters.
   real(R_P)                                        :: t(0:3)          !< Phase marks (wtime).
   character(*), parameter :: HERE='elisha_spectral_solver_object%solve_blocks' !< For the messages.

   if (.not.self%is_initialized) then ! every rank alike: the setup is collective
      call elisha_set_error(stat, errmsg, ELISHA_ERR_NOT_INITIALIZED, &
                            'elisha_spectral_solver_object%solve_blocks: called before setup')
      return
   endif
   if (self%redundant) then
      if (present(energy)) then ! every rank alike
         call elisha_set_error(stat, errmsg, ELISHA_ERR_ENERGY_REDUNDANT, &
                               'elisha_spectral_solver_object%solve_blocks: no energy with redundant')
         return
      endif
      call self%solve_everywhere(nbl=nbl, coords=coords, buf=buf, factor=factor, gw=gw, out=out, stat=stat, &
                                 errmsg=errmsg)
      return
   endif
   if (self%slab) then
      call self%solve_slabs(nbl=nbl, coords=coords, buf=buf, factor=factor, gw=gw, out=out, stat=stat, errmsg=errmsg, &
                            energy=energy)
      return
   endif
   t(0) = self%comm%wtime()
   ub     = self%nijk + gw
   ncell  = product(self%nijk)
   nghost = product(self%nijk + 2 * gw)
   nr     = self%comm%nranks
   stat = ELISHA_OK
   if (size(buf) < nbl*ncell) &
      call elisha_set_error(stat, errmsg, ELISHA_ERR_SHAPE, HERE//': the packed source is too short')
   if (size(out) < nbl*nghost) &
      call elisha_set_error(stat, errmsg, ELISHA_ERR_SHAPE, HERE//': the packed output is too short')
   call agree_status(self%comm, stat, errmsg, HERE)
   if (stat /= ELISHA_OK) return
   ! gather to rank 0
   allocate(counts(nr), displs(nr), rcounts(nr), rdispls(nr))
   counts = 0 ; displs = 0 ; rcounts = 0 ; rdispls = 0
   call self%comm%gather_i4(nbl, counts, stat)
   if (collective_failed(stat, errmsg, HERE)) return
   blocks_total = 0
   if (self%comm%rank == 0) then
      do r=2, nr
         displs(r) = displs(r-1) + counts(r-1)
      enddo
      blocks_total = sum(counts)
      rcounts = 3 * counts ; rdispls = 3 * displs
   endif
   allocate(coords_all(3,max(blocks_total,1)), buf_all(max(blocks_total*nghost,blocks_total*ncell,1)))
   call self%comm%gatherv_i4(coords, 3*nbl, coords_all, rcounts, rdispls, stat)
   if (collective_failed(stat, errmsg, HERE)) return
   if (self%comm%rank == 0) then
      rcounts = ncell * counts ; rdispls = ncell * displs
   endif
   call self%comm%gatherv_r(buf, nbl*ncell, buf_all, rcounts, rdispls, stat)
   if (collective_failed(stat, errmsg, HERE)) return
   t(1) = self%comm%wtime()
   ! rank 0: place, transform, pack every block with its ghosts by the periodic wrap
   e = 0._R_P
   if (self%comm%rank == 0) then
      if (self%isolated .or. self%extended) self%f_global = 0._R_P ! the padding (the images) must hold no source
      c = 0
      do b=1, blocks_total
         do k=1, self%nijk(3)
            do j=1, self%nijk(2)
               do i=1, self%nijk(1)
                  c = c + 1
                  self%f_global(coords_all(1,b)*self%nijk(1)+i, coords_all(2,b)*self%nijk(2)+j, &
                                coords_all(3,b)*self%nijk(3)+k) = buf_all(c)
               enddo
            enddo
         enddo
      enddo
      if (self%extended) call self%extend_global
      if (present(energy)) self%f_source = self%f_global
      if (self%free) call self%free_ghosts(factor=factor, gd=max(gw, self%dg_order))
      if (self%pcg) then
         call self%solve_pcg(factor=factor, stat=stat, errmsg=errmsg)
      else
         call self%transform(factor=factor, stat=stat, errmsg=errmsg)
      endif
      if (present(energy)) then
         if (self%extended) then ! the interior only: the images are not cells
            e = sum(self%f_source(1:self%n(1),1:self%n(2),1:self%n(3)) * self%f_global(1:self%n(1),1:self%n(2),1:self%n(3)))
         else
            e = sum(self%f_source * self%f_global)
         endif
      endif
      if (self%pcg) then ! the ghosts by the faces' rules (no extension holds them)
         allocate(u_pad(1-gw:self%n(1)+gw,1-gw:self%n(2)+gw,1-gw:self%n(3)+gw))
         u_pad = 0._R_P
         u_pad(1:self%n(1),1:self%n(2),1:self%n(3)) = self%f_global
         call self%fill_ghosts(u_pad, gw)
         if (self%free) call self%put_free_ghosts(u_pad, gw)
      endif
      c = 0
      do b=1, blocks_total
         do k=1-gw, ub(3)
            g(3) = modulo(coords_all(3,b)*self%nijk(3) + k - 1, self%m(3)) + 1
            do j=1-gw, ub(2)
               g(2) = modulo(coords_all(2,b)*self%nijk(2) + j - 1, self%m(2)) + 1
               do i=1-gw, ub(1)
                  g(1) = modulo(coords_all(1,b)*self%nijk(1) + i - 1, self%m(1)) + 1
                  c = c + 1
                  if (self%pcg) then
                     buf_all(c) = u_pad(coords_all(1,b)*self%nijk(1)+i, coords_all(2,b)*self%nijk(2)+j, &
                                        coords_all(3,b)*self%nijk(3)+k)
                  else
                     buf_all(c) = self%f_global(g(1),g(2),g(3))
                  endif
               enddo
            enddo
         enddo
      enddo
      rcounts = nghost * counts ; rdispls = nghost * displs
   endif
   ! the rank-0 solve fails on rank 0 only (no FFT library, a backend's failure): every rank takes its status
   call agree_status(self%comm, stat, errmsg, HERE)
   if (stat /= ELISHA_OK) return
   if (self%pcg) then ! the CG's convergence, rank 0's, to every rank (ELISHA notes E1c, decision C)
      nc_ = 0
      if (self%comm%rank == 0 .and. .not.self%converged) nc_ = 1
      call self%comm%max_i4(nc_, stat)
      if (collective_failed(stat, errmsg, HERE)) return
      self%converged = nc_(1) == 0
   endif
   t(2) = self%comm%wtime()
   ! scatter back, ghosts included
   call self%comm%scatterv_r(buf_all, rcounts, rdispls, out, nbl*nghost, stat)
   if (collective_failed(stat, errmsg, HERE)) return
   if (present(energy)) then
      e_(1) = e
      call self%comm%bcast_r(e_, 1_I4P, stat)
      if (collective_failed(stat, errmsg, HERE)) return
      energy = e_(1)
   endif
   t(3) = self%comm%wtime()
   self%wall = self%wall + (t(1:3) - t(0:2))
   ! ELISHA notes E1c, decision C: the CG of the wide D(G) stopped at max_iterations above its tolerance (agreed over
   ! the ranks above), `out` the last iterate with its ghosts filled
   if (.not.self%converged) &
      call elisha_set_error(stat, errmsg, ELISHA_ERR_NOT_CONVERGED, HERE//': the CG of the wide D(G) reached '// &
         'max_iterations ('//trim(str(self%max_iterations, .true.))//') above its tolerance ('// &
         trim(str(self%tolerance))//'); rank 0 holds its iterations and residual')
   endsubroutine solve_blocks

   subroutine solve_everywhere(self, nbl, coords, buf, factor, gw, out, stat, errmsg)
   !< `solve_blocks` with `redundant` (issue #84, E5c): the source of every rank gathered to every rank, placed and
   !< transformed on every rank (the same arrays, the same plans: the same values as rank 0 computes in
   !< `solve_blocks`), and each rank's blocks packed from its own global solution. `wall`: gather, solve, 0.
   class(elisha_spectral_solver_object), intent(inout) :: self          !< The solver.
   integer(I4P),              intent(in)    :: nbl           !< Blocks of this rank.
   integer(I4P),              intent(in)    :: coords(1:,1:) !< Block coordinates [3,nbl].
   real(R_P),                 intent(in)    :: buf(1:)       !< Packed interior source of this rank.
   real(R_P),                 intent(in)    :: factor        !< Source factor.
   integer(I4P),              intent(in)    :: gw            !< Ghost layers of `out`.
   real(R_P),                 intent(inout) :: out(1:)       !< Packed potential, ghosts included.
   integer(I4P),              intent(out)   :: stat          !< Status.
   character(:), allocatable, intent(out), optional :: errmsg !< Message.
   integer(I4P)                             :: ncell, nr     !< Interior cells per block, ranks.
   integer(I4P), allocatable                :: counts(:), displs(:), rcounts(:), rdispls(:) !< Allgather layout.
   integer(I4P), allocatable                :: coords_all(:,:) !< Block coordinates of every rank.
   real(R_P),    allocatable                :: buf_all(:)    !< Source of every rank.
   integer(I4P)                             :: ub(3), g(3)   !< Upper bounds, wrapped global index.
   integer(I4P)                             :: b, i, j, k, c, r, blocks_total !< Counters.
   real(R_P)                                :: t(0:2)        !< Phase marks (wtime).
   character(*), parameter :: HERE='elisha_spectral_solver_object%solve_everywhere' !< For the messages.

   t(0) = self%comm%wtime()
   ub    = self%nijk + gw
   ncell = product(self%nijk)
   nr    = self%comm%nranks
   allocate(counts(nr), displs(nr), rcounts(nr), rdispls(nr))
   call self%comm%allgather_i4(nbl, counts, stat)
   if (collective_failed(stat, errmsg, HERE)) return
   displs(1) = 0
   do r=2, nr
      displs(r) = displs(r-1) + counts(r-1)
   enddo
   blocks_total = sum(counts)
   allocate(coords_all(3,max(blocks_total,1)), buf_all(max(blocks_total*ncell,1)))
   rcounts = 3 * counts ; rdispls = 3 * displs
   call self%comm%allgatherv_layout_i4(coords, 3*nbl, coords_all, rcounts, rdispls, stat)
   if (collective_failed(stat, errmsg, HERE)) return
   rcounts = ncell * counts ; rdispls = ncell * displs
   call self%comm%allgatherv_layout_r(buf, nbl*ncell, buf_all, rcounts, rdispls, stat)
   if (collective_failed(stat, errmsg, HERE)) return
   t(1) = self%comm%wtime()
   if (self%extended) self%f_global = 0._R_P
   c = 0
   do b=1, blocks_total
      do k=1, self%nijk(3)
         do j=1, self%nijk(2)
            do i=1, self%nijk(1)
               c = c + 1
               self%f_global(coords_all(1,b)*self%nijk(1)+i, coords_all(2,b)*self%nijk(2)+j, &
                             coords_all(3,b)*self%nijk(3)+k) = buf_all(c)
            enddo
         enddo
      enddo
   enddo
   if (self%extended) call self%extend_global
   call self%transform(factor=factor, stat=stat, errmsg=errmsg) ; if (stat /= ELISHA_OK) return ! every rank alike
   c = 0
   do b=displs(self%comm%rank+1)+1, displs(self%comm%rank+1)+nbl
      do k=1-gw, ub(3)
         g(3) = modulo(coords_all(3,b)*self%nijk(3) + k - 1, self%m(3)) + 1
         do j=1-gw, ub(2)
            g(2) = modulo(coords_all(2,b)*self%nijk(2) + j - 1, self%m(2)) + 1
            do i=1-gw, ub(1)
               g(1) = modulo(coords_all(1,b)*self%nijk(1) + i - 1, self%m(1)) + 1
               c = c + 1
               out(c) = self%f_global(g(1),g(2),g(3))
            enddo
         enddo
      enddo
   enddo
   t(2) = self%comm%wtime()
   self%wall(1:2) = self%wall(1:2) + (t(1:2) - t(0:1))
   endsubroutine solve_everywhere

   subroutine slab_setup(self, stat, errmsg)
   !< The distributed slab transform (issue #84, E5d): rank r owns the z-planes [zr(r), zr(r+1)) of the transform array
   !< and, after the transpose, the y-rows [yr(r), yr(r+1)) of its z-lines; the per-axis maps from a target cell of the
   !< transform array to its source cell and sign (the extensions of `extend_global`, the isolated padding); the
   !< arrays, the plans and, isolated, the kernel's spectrum in the pencil layout.
   class(elisha_spectral_solver_object), intent(inout)         :: self   !< The solver.
   integer(I4P),                         intent(out)           :: stat   !< Status (every rank alike).
   character(:), allocatable,            intent(out), optional :: errmsg !< Message.
   integer(I4P)                             :: nr, me, r, d, t, mm, nzl, nyl !< Counters, sizes.

   nr = self%comm%nranks ; me = self%comm%rank
   self%h1 = self%m(1) / 2 + 1
   allocate(self%zr(0:nr), self%yr(0:nr), self%zown(0:self%m(3)-1))
   do r=0, nr
      self%zr(r) = int((int(r, I8P) * self%m(3)) / nr, I4P)
      self%yr(r) = int((int(r, I8P) * self%m(2)) / nr, I4P)
   enddo
   do r=0, nr-1
      if (self%zr(r+1) > self%zr(r)) self%zown(self%zr(r):self%zr(r+1)-1) = r
   enddo
   mm = maxval(self%m)
   allocate(self%tmap(0:mm-1,3), self%tsgn(0:mm-1,3))
   self%tmap = -1 ; self%tsgn = 0._R_P
   do d=1, 3
      do t=0, self%m(d)-1
         call elisha_axis_map(self%n(d), self%m(d), self%bc(:,d), self%isolated, t, self%tmap(t,d), self%tsgn(t,d))
      enddo
   enddo
   nzl = self%zr(me+1) - self%zr(me) ; nyl = self%yr(me+1) - self%yr(me)
   allocate(self%slab_r(self%m(1),self%m(2),max(nzl,1)), self%slab_c(self%h1,self%m(2),max(nzl,1)), &
            self%pen(self%m(3),self%h1,max(nyl,1)))
   self%slab_r = 0._R_P ; self%slab_c = (0._R_P, 0._R_P) ; self%pen = (0._R_P, 0._R_P)
   call self%slab_make_plans(stat, errmsg)
   ! the plans may fail on some ranks only: every rank takes the worst status before the kernel's collectives
   call agree_status(self%comm, stat, errmsg, 'elisha_spectral_solver_object%slab_setup')
   if (stat /= ELISHA_OK) return
   if (self%isolated) call slab_kernel
   contains
      subroutine slab_kernel
      !< Isolated: the kernel `h^2 G(o)` on this rank's planes (offsets wrapped into (-m/2, m/2]), transformed as the
      !< source is, its spectrum over prod(m) kept in the pencil layout (real: the kernel is even).
      integer(I4P) :: o(3), i, j, k !< Offset, counters.

      do k=1, nzl
         o(3) = self%zr(me) + k - 1 ; if (o(3) > self%m(3) / 2) o(3) = o(3) - self%m(3)
         do j=1, self%m(2)
            o(2) = j - 1 ; if (o(2) > self%m(2) / 2) o(2) = o(2) - self%m(2)
            do i=1, self%m(1)
               o(1) = i - 1 ; if (o(1) > self%m(1) / 2) o(1) = o(1) - self%m(1)
               self%slab_r(i,j,k) = self%dxyz(1)**2 * elisha_lattice_green(o)
            enddo
         enddo
      enddo
      call self%slab_update(to_device=.true.)
      call self%slab_planes(forward=.true., stat=stat, errmsg=errmsg) ; if (stat /= ELISHA_OK) return
      call self%slab_transpose(forward=.true., stat=stat, errmsg=errmsg) ; if (stat /= ELISHA_OK) return
      call self%slab_lines(forward=.true., stat=stat, errmsg=errmsg) ; if (stat /= ELISHA_OK) return
      allocate(self%khat_pen(self%m(3),self%h1,max(nyl,1)))
      call self%slab_keep_kernel
      self%slab_r = 0._R_P
      endsubroutine slab_kernel
   endsubroutine slab_setup

   subroutine solve_slabs(self, nbl, coords, buf, factor, gw, out, stat, errmsg, energy)
   !< `solve_blocks` with `slab` (issue #84, E5d): every block plane sent to the owners of the target planes that map to
   !< it, each rank's planes built (the extensions and the padding by the maps), transformed (`slab_transform`), and
   !< every block's potential with its ghost layers fetched back from the owners of its planes (wrapped modulo m, as
   !< `solve_blocks` reads `f_global`). The arithmetic of every value does not depend on the rank count (one plan per
   !< plane and per row of lines). `wall`: redistribution, transform, return.
   class(elisha_spectral_solver_object), intent(inout)         :: self          !< The solver.
   integer(I4P),              intent(in)            :: nbl           !< Blocks of this rank.
   integer(I4P),              intent(in)            :: coords(1:,1:) !< Block coordinates [3,nbl].
   real(R_P),                 intent(in)            :: buf(1:)       !< Packed interior source of this rank.
   real(R_P),                 intent(in)            :: factor        !< Source factor.
   integer(I4P),              intent(in)            :: gw            !< Ghost layers of `out`.
   real(R_P),                 intent(inout)         :: out(1:)       !< Packed potential, ghosts included.
   integer(I4P),              intent(out)           :: stat          !< Status.
   character(:), allocatable, intent(out), optional :: errmsg        !< Message.
   real(R_P),                 intent(out), optional :: energy        !< sum_cells f phi (all ranks).
   integer(I4P), allocatable :: tgo(:,:), ntgo(:)          !< Owner ranks of the targets of each source z-plane.
   integer(I4P), allocatable :: srcidx(:)                  !< Source z-plane -> local index in src (0: not needed).
   real(R_P),    allocatable :: src(:,:,:)                 !< This rank's source planes [n(1),n(2),needed].
   integer(I4P), allocatable :: sc(:), sd(:), rc(:), rd(:) !< Alltoallv counts and displacements (items).
   integer(I4P), allocatable :: shdr(:), rhdr(:), fill(:)  !< Headers, cursors.
   real(R_P),    allocatable :: sval(:), rval(:)           !< Values.
   integer(I4P)              :: nr, me, nb(3), np_, w, nzl, nsrc, b, i, j, k, t, t1, t2, t3, s1, s2, s3, r, c, e, q
   real(R_P)                 :: sg3, sg23, el, tm(0:3)     !< Signs, energy, phase marks.
   real(R_P)                 :: el_(1)                     !< el, as the reduction's buffer.
   character(*), parameter   :: HERE='elisha_spectral_solver_object%solve_slabs' !< For the messages.

   stat = ELISHA_OK
   tm(0) = self%comm%wtime()
   nr = self%comm%nranks ; me = self%comm%rank
   nb = self%nijk ; np_ = nb(1) * nb(2)
   nzl = self%zr(me+1) - self%zr(me)
   allocate(sc(0:nr-1), sd(0:nr-1), rc(0:nr-1), rd(0:nr-1), fill(0:nr-1))
   ! the owner ranks of the targets of every source plane (1, 2 or 4 targets, deduplicated)
   allocate(tgo(4,0:self%n(3)-1), ntgo(0:self%n(3)-1))
   ntgo = 0
   do t=0, self%m(3)-1
      s3 = self%tmap(t,3)
      if (s3 < 0) cycle
      r = self%zown(t)
      if (any(tgo(1:ntgo(s3),s3) == r)) cycle
      ntgo(s3) = ntgo(s3) + 1 ; tgo(ntgo(s3),s3) = r
   enddo
   ! this rank's source planes (those its target planes map to)
   allocate(srcidx(0:self%n(3)-1))
   srcidx = 0 ; nsrc = 0
   do t=self%zr(me), self%zr(me+1)-1
      s3 = self%tmap(t,3)
      if (s3 < 0) cycle
      if (srcidx(s3) > 0) cycle
      nsrc = nsrc + 1 ; srcidx(s3) = nsrc
   enddo
   allocate(src(self%n(1),self%n(2),max(nsrc,1)))
   ! forward: every block plane (header c1, c2, s3 and its np_ values) to each other owner that needs it, this rank's
   ! own straight into src
   sc = 0
   do b=1, nbl
      do k=1, nb(3)
         s3 = coords(3,b) * nb(3) + k - 1
         do q=1, ntgo(s3)
            if (tgo(q,s3) /= me) sc(tgo(q,s3)) = sc(tgo(q,s3)) + 1
         enddo
      enddo
   enddo
   call exchange_counts
   if (collective_failed(stat, errmsg, HERE)) return
   allocate(shdr(max(3*sum(sc),1)), sval(max(np_*sum(sc),1)), rhdr(max(3*sum(rc),1)), rval(max(np_*sum(rc),1)))
   fill = sd
   do b=1, nbl
      do k=1, nb(3)
         s3 = coords(3,b) * nb(3) + k - 1
         e = (b-1)*np_*nb(3) + (k-1)*np_
         do q=1, ntgo(s3)
            r = tgo(q,s3)
            if (r == me) then
               call put_plane(coords(1,b), coords(2,b), srcidx(s3), buf(e+1:e+np_))
            else
               c = fill(r) ; fill(r) = fill(r) + 1
               shdr(3*c+1:3*c+3) = [coords(1,b), coords(2,b), s3]
               sval(np_*c+1:np_*c+np_) = buf(e+1:e+np_)
            endif
         enddo
      enddo
   enddo
   if (nr > 1) then
      call self%comm%alltoallv_i4(shdr, 3*sc, 3*sd, rhdr, 3*rc, 3*rd, stat)
      if (collective_failed(stat, errmsg, HERE)) return
      call self%comm%alltoallv_r(sval, np_*sc, np_*sd, rval, np_*rc, np_*rd, stat)
      if (collective_failed(stat, errmsg, HERE)) return
   endif
   do c=0, sum(rc)-1
      call put_plane(rhdr(3*c+1), rhdr(3*c+2), srcidx(rhdr(3*c+3)), rval(np_*c+1:np_*c+np_))
   enddo
   ! this rank's target planes by the maps
   do k=1, nzl
      t3 = self%zr(me) + k - 1
      s3 = self%tmap(t3,3) ; sg3 = self%tsgn(t3,3)
      if (s3 < 0) then
         self%slab_r(:,:,k) = 0._R_P
         cycle
      endif
      do t2=0, self%m(2)-1
         s2 = self%tmap(t2,2) ; sg23 = sg3 * self%tsgn(t2,2)
         do t1=0, self%m(1)-1
            s1 = self%tmap(t1,1)
            if (s1 < 0 .or. s2 < 0) then
               self%slab_r(t1+1,t2+1,k) = 0._R_P
            else
               self%slab_r(t1+1,t2+1,k) = sg23 * self%tsgn(t1,1) * src(s1+1,s2+1,srcidx(s3))
            endif
         enddo
      enddo
   enddo
   tm(1) = self%comm%wtime()
   call self%slab_transform(factor=factor, stat=stat, errmsg=errmsg) ; if (stat /= ELISHA_OK) return ! every rank alike
   tm(2) = self%comm%wtime()
   if (present(energy)) then ! the domain's cells only: the images and the padding are not cells
      el = 0._R_P
      do k=1, nzl
         t3 = self%zr(me) + k - 1
         if (t3 >= self%n(3)) cycle
         do j=1, self%n(2)
            do i=1, self%n(1)
               el = el + src(i,j,srcidx(t3)) * self%slab_r(i,j,k)
            enddo
         enddo
      enddo
      el_(1) = el
      call self%comm%sum_r(el_, stat)
      if (collective_failed(stat, errmsg, HERE)) return
      energy = el_(1)
   endif
   ! return: each block's planes, ghosts included, from their other owners (requests c1, c2, t3; windows wrapped
   ! modulo m), this rank's own straight from slab_r
   w = (nb(1) + 2 * gw) * (nb(2) + 2 * gw)
   sc = 0
   do b=1, nbl
      do k=1-gw, nb(3)+gw
         t3 = modulo(coords(3,b) * nb(3) + k - 1, self%m(3))
         if (self%zown(t3) /= me) sc(self%zown(t3)) = sc(self%zown(t3)) + 1
      enddo
   enddo
   call exchange_counts
   if (collective_failed(stat, errmsg, HERE)) return
   deallocate(shdr, rhdr, sval, rval)
   allocate(shdr(max(3*sum(sc),1)), rhdr(max(3*sum(rc),1)), sval(max(w*sum(rc),1)), rval(max(w*sum(sc),1)))
   fill = sd
   do b=1, nbl
      do k=1-gw, nb(3)+gw
         t3 = modulo(coords(3,b) * nb(3) + k - 1, self%m(3))
         r = self%zown(t3)
         if (r == me) cycle
         c = fill(r) ; fill(r) = fill(r) + 1
         shdr(3*c+1:3*c+3) = [coords(1,b), coords(2,b), t3]
      enddo
   enddo
   if (nr > 1) then
      call self%comm%alltoallv_i4(shdr, 3*sc, 3*sd, rhdr, 3*rc, 3*rd, stat)
      if (collective_failed(stat, errmsg, HERE)) return
   endif
   do c=0, sum(rc)-1
      call get_plane(rhdr(3*c+1), rhdr(3*c+2), rhdr(3*c+3) - self%zr(me) + 1, sval(w*c+1:w*c+w))
   enddo
   if (nr > 1) then
      call self%comm%alltoallv_r(sval, w*rc, w*rd, rval, w*sc, w*sd, stat)
      if (collective_failed(stat, errmsg, HERE)) return
   endif
   fill = sd
   e = 0
   do b=1, nbl
      do k=1-gw, nb(3)+gw
         t3 = modulo(coords(3,b) * nb(3) + k - 1, self%m(3))
         r = self%zown(t3)
         if (r == me) then
            call get_plane(coords(1,b), coords(2,b), t3 - self%zr(me) + 1, out(e+1:e+w))
         else
            c = fill(r) ; fill(r) = fill(r) + 1
            out(e+1:e+w) = rval(w*c+1:w*c+w)
         endif
         e = e + w
      enddo
   enddo
   tm(3) = self%comm%wtime()
   self%wall = self%wall + (tm(1:3) - tm(0:2))
   contains
      subroutine put_plane(c1, c2, l, v)
      !< The interior plane v of the block (c1, c2) into the local source plane l.
      integer(I4P), intent(in) :: c1, c2, l !< Block coordinates, local source plane.
      real(R_P),    intent(in) :: v(:)      !< Values, i fastest.
      integer(I4P)             :: jj        !< Counter.

      do jj=1, nb(2)
         src(c1*nb(1)+1:c1*nb(1)+nb(1),c2*nb(2)+jj,l) = v((jj-1)*nb(1)+1:jj*nb(1))
      enddo
      endsubroutine put_plane

      subroutine get_plane(c1, c2, l, v)
      !< The window of the block (c1, c2), ghosts included and wrapped modulo m, from the local plane l of slab_r.
      integer(I4P), intent(in)  :: c1, c2, l      !< Block coordinates, local plane.
      real(R_P),    intent(out) :: v(:)           !< Values, i fastest.
      integer(I4P)              :: ii, jj, o, wx, i0, y !< Counters, row width, first column (0-based), row.

      wx = nb(1) + 2 * gw ; i0 = c1 * nb(1) - gw
      o = 0
      do jj=1-gw, nb(2)+gw
         y = modulo(c2 * nb(2) + jj - 1, self%m(2)) + 1
         if (i0 >= 0 .and. i0 + wx <= self%m(1)) then
            v(o+1:o+wx) = self%slab_r(i0+1:i0+wx,y,l)
         else
            do ii=1, wx
               v(o+ii) = self%slab_r(modulo(i0 + ii - 1, self%m(1)) + 1,y,l)
            enddo
         endif
         o = o + wx
      enddo
      endsubroutine get_plane

      subroutine exchange_counts
      !< The receive counts from the send counts, and both displacements (items); `stat` of the exchange.
      call self%comm%alltoall_i4(sc, rc, stat)
      sd(0) = 0 ; rd(0) = 0
      do r=1, nr-1
         sd(r) = sd(r-1) + sc(r-1) ; rd(r) = rd(r-1) + rc(r-1)
      enddo
      endsubroutine exchange_counts
   endsubroutine solve_slabs

   subroutine slab_transform(self, factor, stat, errmsg)
   !< `slab_r <- factor L^-1 slab_r` on the distributed slabs: the planes' 2-D transforms, the transpose to z-lines,
   !< their 1-D transforms, the spectral solve, and back; the stages run where the backend keeps the slab arrays
   !< (`slab_update` moves `slab_r` there and back).
   class(elisha_spectral_solver_object), intent(inout) :: self    !< The solver.
   real(R_P),                 intent(in)    :: factor  !< Source factor.
   integer(I4P),              intent(out)   :: stat    !< Status.
   character(:), allocatable, intent(out), optional :: errmsg !< Message.

   call self%slab_update(to_device=.true.)
   call self%slab_planes(forward=.true., stat=stat, errmsg=errmsg) ; if (stat /= ELISHA_OK) return
   call self%slab_transpose(forward=.true., stat=stat, errmsg=errmsg) ; if (stat /= ELISHA_OK) return
   call self%slab_lines(forward=.true., stat=stat, errmsg=errmsg) ; if (stat /= ELISHA_OK) return
   call self%slab_spectral(factor=factor)
   call self%slab_lines(forward=.false., stat=stat, errmsg=errmsg) ; if (stat /= ELISHA_OK) return
   call self%slab_transpose(forward=.false., stat=stat, errmsg=errmsg) ; if (stat /= ELISHA_OK) return
   call self%slab_planes(forward=.false., stat=stat, errmsg=errmsg) ; if (stat /= ELISHA_OK) return
   call self%slab_update(to_device=.false.)
   endsubroutine slab_transform

   subroutine global_potential(self, g, stat)
   !< Rank 0: the transform array after the last solve, [m(1),m(2),m(3)] (the slabs gathered; `f_global` otherwise).
   !< For diagnostics; other ranks get an empty array (or a 1-cell one on the slab path). Collective on the slab path.
   class(elisha_spectral_solver_object), intent(in)    :: self     !< The solver.
   real(R_P), allocatable,    intent(inout) :: g(:,:,:) !< The array.
   integer(I4P),              intent(out)   :: stat     !< Status.
   integer(I4P), allocatable :: rc(:), rd(:)             !< Counts and displacements.
   integer(I4P)              :: nr, r, nzl               !< Counters.

   stat = ELISHA_OK
   if (allocated(g)) deallocate(g)
   if (.not.self%slab) then
      if (self%comm%rank == 0) then
         g = self%f_global
      else
         allocate(g(0,0,0))
      endif
      return
   endif
   nr = self%comm%nranks
   allocate(rc(0:nr-1), rd(0:nr-1))
   do r=0, nr-1
      rc(r) = self%m(1) * self%m(2) * (self%zr(r+1) - self%zr(r))
      rd(r) = self%m(1) * self%m(2) * self%zr(r)
   enddo
   if (self%comm%rank == 0) then
      allocate(g(self%m(1),self%m(2),self%m(3)))
   else
      allocate(g(1,1,1))
   endif
   nzl = self%zr(self%comm%rank+1) - self%zr(self%comm%rank)
   call self%comm%gatherv_r(self%slab_r, self%m(1)*self%m(2)*nzl, g, rc, rd, stat)
   endsubroutine global_potential

   ! backend hooks
   ! private methods
   subroutine make_kernel(self, stat, errmsg)
   !< Rank 0, isolated: place the kernel `h^2 G(o)` at every offset `o`, wrapped into `(-m/2, m/2]` per direction, and
   !< keep its spectrum over `prod(m)` (`transform_kernel`).
   class(elisha_spectral_solver_object), intent(inout)         :: self   !< The solver.
   integer(I4P),                         intent(out)           :: stat   !< Status.
   character(:), allocatable,            intent(out), optional :: errmsg !< Message.
   integer(I4P)                             :: o(3)    !< Offset.
   integer(I4P)                             :: i, j, k !< Counters.

   do k=1, self%m(3)
      o(3) = k - 1 ; if (o(3) > self%m(3) / 2) o(3) = o(3) - self%m(3)
      do j=1, self%m(2)
         o(2) = j - 1 ; if (o(2) > self%m(2) / 2) o(2) = o(2) - self%m(2)
         do i=1, self%m(1)
            o(1) = i - 1 ; if (o(1) > self%m(1) / 2) o(1) = o(1) - self%m(1)
            self%f_global(i,j,k) = self%dxyz(1)**2 * elisha_lattice_green(o)
         enddo
      enddo
   enddo
   allocate(self%khat(self%m(1)/2+1,self%m(2),self%m(3)))
   call self%transform_kernel(stat, errmsg) ; if (stat /= ELISHA_OK) return
   self%f_global = 0._R_P
   endsubroutine make_kernel

   ! backend hooks
   subroutine destroy_plans(self)
   !< Destroy the FFTW plans.
   class(elisha_spectral_solver_object), intent(inout) :: self !< The solver.

#ifdef _FFTW
   if (c_associated(self%plan_r2c)) call fftw_destroy_plan(self%plan_r2c)
   if (c_associated(self%plan_c2r)) call fftw_destroy_plan(self%plan_c2r)
#endif
   self%plan_r2c = c_null_ptr
   self%plan_c2r = c_null_ptr
   endsubroutine destroy_plans

   subroutine make_plans(self, stat, errmsg)
   !< Make the FFTW plans on rank 0's global arrays (FFTW_ESTIMATE: the arrays are not touched, the plans deterministic).
   class(elisha_spectral_solver_object), intent(inout)         :: self   !< The solver.
   integer(I4P),                         intent(out)           :: stat   !< Status.
   character(:), allocatable,            intent(out), optional :: errmsg !< Message.

#ifdef _FFTW
   ! FFTW is row-major: the slowest dimension first
   self%plan_r2c = fftw_plan_dft_r2c_3d(int(self%m(3), c_int), int(self%m(2), c_int), int(self%m(1), c_int), &
                                        self%f_global, self%f_hat, FFTW_ESTIMATE)
   self%plan_c2r = fftw_plan_dft_c2r_3d(int(self%m(3), c_int), int(self%m(2), c_int), int(self%m(1), c_int), &
                                        self%f_hat, self%f_global, FFTW_ESTIMATE)
   stat = ELISHA_OK
   if (.not.c_associated(self%plan_r2c) .or. .not.c_associated(self%plan_c2r)) &
      call elisha_set_error(stat, errmsg, ELISHA_ERR_FFT_PLAN, &
                            'elisha_spectral_solver_object%make_plans: FFTW could not make the plans')
#else
   call elisha_set_error(stat, errmsg, ELISHA_ERR_NO_FFT, 'elisha_spectral_solver_object%make_plans: '// &
                         'built without an FFT library (build with -D_FFTW and link fftw3)')
#endif
   endsubroutine make_plans

   pure function has_device_redundant(self) result(yes)
   !< True when `solve_redundant_device` solves on device arrays (a device backend, the redundant path); here never.
   class(elisha_spectral_solver_object), intent(in) :: self !< The solver.
   logical                                          :: yes  !< The device path exists.

   yes = .false. .and. self%redundant
   endfunction has_device_redundant

   subroutine solve_redundant_device(self, nbl, coords, nb, buf, nout, out, factor, stat, errmsg)
   !< `solve_everywhere` (gw = 0) on device-resident `buf` and `out` (M7, audit C4: the hierarchical solver's root on
   !< the device). The host solver has none: `ELISHA_ERR_ARGUMENT` (a device backend overrides it; the caller asks
   !< `has_device_redundant` first).
   class(elisha_spectral_solver_object), intent(inout)         :: self          !< The solver.
   integer(I4P),                         intent(in)            :: nbl           !< Blocks of this rank.
   integer(I4P),                         intent(in)            :: coords(1:,1:) !< Block coordinates [3,nbl].
   integer(I4P),                         intent(in)            :: nb            !< Size of buf.
   real(R_P),                            intent(in)            :: buf(nb)       !< Packed source (on the device).
   integer(I4P),                         intent(in)            :: nout          !< Size of out.
   real(R_P),                            intent(inout)         :: out(nout)     !< Packed potential (on the device).
   real(R_P),                            intent(in)            :: factor        !< Source factor.
   integer(I4P),                         intent(out)           :: stat          !< Status.
   character(:), allocatable,            intent(out), optional :: errmsg        !< Message.

   if (.false.) out(1) = buf(1) * factor + nbl + coords(1,1)
   call elisha_set_error(stat, errmsg, ELISHA_ERR_ARGUMENT, 'elisha_spectral_solver_object%solve_redundant_device: '// &
                         'the host solver has no device arrays')
   endsubroutine solve_redundant_device

   subroutine stage_source(self, buf, bufh, stat)
   !< `solve_packed(on_device)`: a host copy of the device source. The host solver has no device arrays:
   !< `ELISHA_ERR_ARGUMENT` (a device backend overrides it).
   class(elisha_spectral_solver_object), intent(inout) :: self    !< The solver.
   real(R_P),                            intent(in)    :: buf(1:) !< Source (on the device).
   real(R_P), allocatable,               intent(out)   :: bufh(:) !< Its host copy.
   integer(I4P),                         intent(out)   :: stat    !< Status.

   if (.false.) bufh = buf ! the host solver: unused
   stat = ELISHA_ERR_ARGUMENT
   endsubroutine stage_source

   subroutine stage_potential(self, phi, to_host)
   !< `solve_packed(on_device)`: phi from the device to the host (`to_host`) or back. Nothing here (the host solver
   !< refuses `on_device` in `stage_source`).
   class(elisha_spectral_solver_object), intent(inout) :: self         !< The solver.
   real(R_P),                            intent(inout) :: phi(:,:,:,:) !< Potential (contiguous).
   logical,                              intent(in)    :: to_host      !< Direction.

   if (.false.) phi(1,1,1,1) = merge(1._R_P, 0._R_P, to_host)
   endsubroutine stage_potential

   subroutine extend_global(self)
   !< Rank 0: the source extended over the m cells of every extended axis (cell index t = 0..m-1, array index t + 1),
   !< one axis after the other (the extensions are separable):
   !<   - ghost-Dirichlet (m = 2 (n + 1)): f(t) for t < n, 0 at t = n and t = m - 1 (the ghost centres), -f(2 n - t)
   !<     for n < t < m - 1;
   !<   - face kinds: f(t) for t < n; the reflection about the high face, s_hi f(2 n - 1 - t), for n <= t < 2 n; with
   !<     m = 4 n (the two faces differ) the reflection of [0, 2 n) about the low face, s_lo g(4 n - 1 - t), for
   !<     2 n <= t < 4 n. The sign s is -1 at a Dirichlet face, +1 at a Neumann one; the periodic extension then has
   !<     both reflection symmetries, and so does its periodic solution: the boundary conditions on [0, n).
   class(elisha_spectral_solver_object), intent(inout) :: self !< The solver.
   integer(I4P) :: t, d_, src   !< Cell index, axis, source cell index.
   real(R_P)    :: sg           !< Reflection sign.

   do d_=1, 3
      if (self%m(d_) == self%n(d_)) cycle ! a null or periodic axis
      do t=self%n(d_), self%m(d_)-1
         if (all(self%bc(:,d_) == ELISHA_BC_DIRICHLET_GHOST)) then
            if (t == self%n(d_) .or. t == self%m(d_)-1) then
               call elisha_set_plane(self%f_global, d_, t, 0, 0._R_P)
            else
               call elisha_set_plane(self%f_global, d_, t, 2*self%n(d_)-t, -1._R_P)
            endif
         elseif (t < 2*self%n(d_)) then
            sg = merge(-1._R_P, 1._R_P, self%bc(2,d_) == ELISHA_BC_DIRICHLET)
            call elisha_set_plane(self%f_global, d_, t, 2*self%n(d_)-1-t, sg)
         else
            sg = merge(-1._R_P, 1._R_P, self%bc(1,d_) == ELISHA_BC_DIRICHLET)
            src = 4*self%n(d_) - 1 - t
            call elisha_set_plane(self%f_global, d_, t, src, sg)
         endif
      enddo
   enddo
   endsubroutine extend_global

   subroutine fill_ghosts(self, u, ng)
   !< Rank 0: the `ng` ghost layers of the padded global array `u` by the faces' rules, one axis after the other:
   !< periodic (and null) the wrap, Dirichlet and Neumann at the face the odd and even reflections about it,
   !< ghost-Dirichlet the odd reflection about the ghost centre (zero there), zero-ghost zero.
   class(elisha_spectral_solver_object), intent(in)    :: self !< The solver.
   integer(I4P),              intent(in)    :: ng   !< Ghost layers.
   real(R_P),                 intent(inout) :: u(1-ng:,1-ng:,1-ng:) !< Padded array [1-ng:n+ng per axis].
   integer(I4P)                             :: d, j, lo, hi, n !< Axis, layer, ghost indices, cells.

   do d=1, 3
      n = self%n(d)
      do j=1, ng
         lo = 1 - j ; hi = n + j
         if (all(self%bc(:,d) == ELISHA_BC_PERIODIC)) then
            call copy_plane(d, lo, modulo(lo - 1, n) + 1, 1._R_P)
            call copy_plane(d, hi, modulo(hi - 1, n) + 1, 1._R_P)
            cycle
         endif
         select case(self%bc(1,d))
         case(ELISHA_BC_DIRICHLET)
            call copy_plane(d, lo, j, -1._R_P)
         case(ELISHA_BC_NEUMANN)
            call copy_plane(d, lo, j, 1._R_P)
         case(ELISHA_BC_DIRICHLET_GHOST)
            if (j == 1) then
               call copy_plane(d, lo, 1, 0._R_P)
            else
               call copy_plane(d, lo, j-1, -1._R_P)
            endif
         case default ! zero ghosts
            call copy_plane(d, lo, 1, 0._R_P)
         endselect
         select case(self%bc(2,d))
         case(ELISHA_BC_DIRICHLET)
            call copy_plane(d, hi, n+1-j, -1._R_P)
         case(ELISHA_BC_NEUMANN)
            call copy_plane(d, hi, n+1-j, 1._R_P)
         case(ELISHA_BC_DIRICHLET_GHOST)
            if (j == 1) then
               call copy_plane(d, hi, n, 0._R_P)
            else
               call copy_plane(d, hi, n+2-j, -1._R_P)
            endif
         case default
            call copy_plane(d, hi, n, 0._R_P)
         endselect
      enddo
   enddo
   contains
      subroutine copy_plane(d_, t, src, sg)
      !< The plane t of axis d_ set to sg times the plane src (sg = 0 zeroes it).
      integer(I4P), intent(in) :: d_, t, src !< Axis, target and source indices.
      real(R_P),    intent(in) :: sg         !< Sign.

      select case(d_)
      case(1)
         u(t,:,:) = sg * u(src,:,:)
      case(2)
         u(:,t,:) = sg * u(:,src,:)
      case(3)
         u(:,:,t) = sg * u(:,:,src)
      endselect
      endsubroutine copy_plane
   endsubroutine fill_ghosts

   subroutine apply_wide(self, x, y)
   !< Rank 0: `y = D(G) x` on the global interior, the ghosts by the faces' rules (`fill_ghosts`).
   class(elisha_spectral_solver_object), intent(in)  :: self          !< The solver.
   real(R_P),                 intent(in)  :: x(1:,1:,1:)   !< Interior values [n].
   real(R_P),                 intent(out) :: y(1:,1:,1:)   !< Result [n].
   real(R_P), allocatable                 :: u(:,:,:)      !< Padded copy.
   integer(I4P)                           :: p, i, j, k, o !< Reach, counters.
   real(R_P)                              :: w(3)          !< 1 / dx^2 per axis (0 along a null one).

   p = self%dg_order
   allocate(u(1-p:self%n(1)+p,1-p:self%n(2)+p,1-p:self%n(3)+p))
   u = 0._R_P
   u(1:self%n(1),1:self%n(2),1:self%n(3)) = x
   call self%fill_ghosts(u, p)
   w = 0._R_P
   where (.not.self%null_xyz) w = 1._R_P / self%dxyz**2
   do k=1, self%n(3)
      do j=1, self%n(2)
         do i=1, self%n(1)
            y(i,j,k) = 0._R_P
            do o=-p, p
               y(i,j,k) = y(i,j,k) + self%stencil(o) * (w(1) * u(i+o,j,k) + w(2) * u(i,j+o,k) + w(3) * u(i,j,k+o))
            enddo
         enddo
      enddo
   enddo
   endsubroutine apply_wide

   subroutine precondition(self, r, z, stat, errmsg)
   !< Rank 0: `z = M^-1 r`, M the wide p = 2 operator with the same faces: the 2^z parity sub-problems, each solved by
   !< its sub-solver's transform (the extension of its kinds, then the division by its eigenvalues).
   class(elisha_spectral_solver_object), intent(inout) :: self        !< The solver.
   real(R_P),                 intent(in)    :: r(1:,1:,1:) !< Residual [n].
   real(R_P),                 intent(inout) :: z(1:,1:,1:) !< Correction [n].
   integer(I4P),              intent(out)   :: stat        !< Status.
   character(:), allocatable, intent(out), optional :: errmsg !< Message.
   integer(I4P)                             :: c, i, j, k, ii, jj, kk !< Counters, global indices.

   stat = ELISHA_OK
   do c=1, self%nsub
      associate(sb=>self%sub(c)%s, par=>self%sub_par(:,c))
      sb%f_global = 0._R_P
      do k=1, sb%n(3)
         kk = merge(k, 2*k-1+par(3), par(3) < 0)
         do j=1, sb%n(2)
            jj = merge(j, 2*j-1+par(2), par(2) < 0)
            do i=1, sb%n(1)
               ii = merge(i, 2*i-1+par(1), par(1) < 0)
               sb%f_global(i,j,k) = r(ii,jj,kk)
            enddo
         enddo
      enddo
      if (sb%extended) call sb%extend_global
      call sb%transform(factor=1._R_P, stat=stat, errmsg=errmsg)
      if (stat /= ELISHA_OK) return
      do k=1, sb%n(3)
         kk = merge(k, 2*k-1+par(3), par(3) < 0)
         do j=1, sb%n(2)
            jj = merge(j, 2*j-1+par(2), par(2) < 0)
            do i=1, sb%n(1)
               ii = merge(i, 2*i-1+par(1), par(1) < 0)
               z(ii,jj,kk) = sb%f_global(i,j,k)
            enddo
         enddo
      enddo
      endassociate
   enddo
   endsubroutine precondition

   subroutine solve_pcg(self, factor, stat, errmsg)
   !< Rank 0: `f_global <- u`, `D(G) u = factor f_global` with zero-ghost faces (E2b), by conjugate gradients
   !< preconditioned by `precondition`, from zero, to `tolerance` (the max-norm residual relative to the source).
   !< D(G) and M are negative definite here (the zero ghosts remove every null mode), which the iteration takes as is.
   class(elisha_spectral_solver_object), intent(inout) :: self !< The solver.
   real(R_P),                 intent(in)    :: factor !< Source factor.
   integer(I4P),              intent(out)   :: stat   !< Status.
   character(:), allocatable, intent(out), optional :: errmsg !< Message.
   real(R_P), allocatable :: f(:,:,:), u(:,:,:), r(:,:,:), z(:,:,:), d(:,:,:), q(:,:,:) !< Vectors.
   real(R_P)              :: fmax, rz, rz_new, alpha !< Scalars.
   integer(I4P)           :: it !< Counter.

   allocate(f, mold=self%f_global)
   f = factor * self%f_global
   if (self%free) call self%subtract_free_ghosts(f) ! D(G)_0 u = factor f - D(G) u_g (E3c)
   allocate(u, r, z, d, q, mold=f)
   u = 0._R_P ; r = f ; z = 0._R_P
   fmax = maxval(abs(f))
   self%iterations = 0 ; self%residual = 0._R_P
   stat = ELISHA_OK
   if (fmax > 0._R_P) then
      call self%precondition(r, z, stat, errmsg) ; if (stat /= ELISHA_OK) return
      d = z
      rz = sum(r * z)
      self%residual = 1._R_P
      do it=1, self%max_iterations
         call self%apply_wide(d, q)
         alpha = rz / sum(d * q)
         u = u + alpha * d
         r = r - alpha * q
         self%iterations = it
         self%residual = maxval(abs(r)) / fmax
         if (self%residual <= self%tolerance) exit
         call self%precondition(r, z, stat, errmsg) ; if (stat /= ELISHA_OK) return
         rz_new = sum(r * z)
         d = z + (rz_new / rz) * d
         rz = rz_new
      enddo
      ! rank 0 only; solve_blocks agrees it over the ranks (ADAM printed a warning here and went on)
      self%converged = .not.(self%residual > self%tolerance)
   endif
   self%f_global = u
   endsubroutine solve_pcg

   subroutine free_ghosts(self, factor, gd)
   !< Rank 0, E3c: `g_free`, to depth `gd`, the ghost slab of every free-space face (the cells outside along that face's
   !< axis, inside along the others): the continuum free-space potential of the gathered source,
   !< `g(x) = -sum_y factor f(y) dV / (4 pi |x - y|)` over the interior cell centres, as PRISM's exact/open face.
   class(elisha_spectral_solver_object), intent(inout) :: self   !< The solver.
   real(R_P),                 intent(in)    :: factor !< Source factor.
   integer(I4P),              intent(in)    :: gd     !< Ghost layers.
   real(R_P)                                :: c      !< -factor dV / (4 pi).
   real(R_P)                                :: s, dz2, dyz2 !< Partial sum, squared distances.
   integer(I4P)                             :: lo(3), hi(3) !< The slab.
   integer(I4P)                             :: d, side, i, j, k, ii, jj, kk !< Counters.

   if (allocated(self%g_free)) deallocate(self%g_free)
   allocate(self%g_free(1-gd:self%n(1)+gd,1-gd:self%n(2)+gd,1-gd:self%n(3)+gd))
   self%g_free = 0._R_P
   c = -factor * product(self%dxyz) / (4._R_P * acos(-1._R_P))
   do d=1, 3
      if (self%null_xyz(d)) cycle
      do side=1, 2
         if (self%bc(side,d) /= ELISHA_BC_FREE_GHOSTS) cycle
         lo = 1 ; hi = self%n
         if (side == 1) then
            lo(d) = 1 - gd ; hi(d) = 0
         else
            lo(d) = self%n(d) + 1 ; hi(d) = self%n(d) + gd
         endif
         do k=lo(3), hi(3)
            do j=lo(2), hi(2)
               do i=lo(1), hi(1)
                  s = 0._R_P
                  do kk=1, self%n(3)
                     dz2 = (real(k - kk, R_P) * self%dxyz(3))**2
                     do jj=1, self%n(2)
                        dyz2 = dz2 + (real(j - jj, R_P) * self%dxyz(2))**2
                        do ii=1, self%n(1)
                           s = s + self%f_global(ii,jj,kk) / sqrt(dyz2 + (real(i - ii, R_P) * self%dxyz(1))**2)
                        enddo
                     enddo
                  enddo
                  self%g_free(i,j,k) = c * s
               enddo
            enddo
         enddo
      enddo
   enddo
   endsubroutine free_ghosts

   subroutine subtract_free_ghosts(self, f)
   !< Rank 0, E3c: `f <- f - D(G) u_g` on the interior, u_g the free-space ghost slabs (`g_free`) and zero inside.
   class(elisha_spectral_solver_object), intent(in)    :: self        !< The solver.
   real(R_P),                 intent(inout) :: f(1:,1:,1:) !< Right-hand side [n].
   integer(I4P)                             :: p, i, j, k, o !< Reach, counters.
   real(R_P)                                :: w(3), y   !< 1 / dx^2 per axis (0 along a null one), D(G) u_g.

   p = self%dg_order
   w = 0._R_P
   where (.not.self%null_xyz) w = 1._R_P / self%dxyz**2
   associate(g=>self%g_free)
   do k=1, self%n(3)
      do j=1, self%n(2)
         do i=1, self%n(1)
            y = 0._R_P
            do o=-p, p
               y = y + self%stencil(o) * (w(1) * g(i+o,j,k) + w(2) * g(i,j+o,k) + w(3) * g(i,j,k+o))
            enddo
            f(i,j,k) = f(i,j,k) - y
         enddo
      enddo
   enddo
   endassociate
   endsubroutine subtract_free_ghosts

   subroutine put_free_ghosts(self, u, ng)
   !< Rank 0, E3c: the `ng` ghost layers of every free-space face of the padded array `u` set to `g_free`.
   class(elisha_spectral_solver_object), intent(in)    :: self !< The solver.
   integer(I4P),              intent(in)    :: ng   !< Ghost layers.
   real(R_P),                 intent(inout) :: u(1-ng:,1-ng:,1-ng:) !< Padded array [1-ng:n+ng per axis].
   integer(I4P)                             :: lo(3), hi(3), d, side !< The slab, counters.

   do d=1, 3
      if (self%null_xyz(d)) cycle
      do side=1, 2
         if (self%bc(side,d) /= ELISHA_BC_FREE_GHOSTS) cycle
         lo = 1 ; hi = self%n
         if (side == 1) then
            lo(d) = 1 - ng ; hi(d) = 0
         else
            lo(d) = self%n(d) + 1 ; hi(d) = self%n(d) + ng
         endif
         u(lo(1):hi(1),lo(2):hi(2),lo(3):hi(3)) = self%g_free(lo(1):hi(1),lo(2):hi(2),lo(3):hi(3))
      enddo
   enddo
   endsubroutine put_free_ghosts

   subroutine transform(self, factor, stat, errmsg)
   !< Rank 0: `f_global <- factor * L_h^-1 f_global`. Periodic: the modes of zero eigenvalue (the mean) set to zero;
   !< isolated: the convolution with the kernel (`khat`).
   class(elisha_spectral_solver_object), intent(inout) :: self    !< The solver.
   real(R_P),                 intent(in)    :: factor  !< Source factor.
   integer(I4P),                         intent(out)           :: stat   !< Status.
   character(:), allocatable,            intent(out), optional :: errmsg !< Message.
   real(R_P)                                :: scale   !< Factor over the unnormalised transform pair.
   real(R_P)                                :: lam     !< Eigenvalue of the mode.
   integer(I4P)                             :: i, j, k !< Counters.

#ifdef _FFTW
   stat = ELISHA_OK
   call fftw_execute_dft_r2c(self%plan_r2c, self%f_global, self%f_hat)
   if (self%isolated) then
      self%f_hat = self%f_hat * (factor * self%khat)
      call fftw_execute_dft_c2r(self%plan_c2r, self%f_hat, self%f_global)
      return
   endif
   scale = factor / real(product(self%nt), R_P)
   do k=1, self%nt(3)
      do j=1, self%nt(2)
         do i=1, self%nt(1)/2+1
            lam = self%lam1(i-1) + self%lam2(j-1) + self%lam3(k-1)
            if (lam == 0._R_P) then
               self%f_hat(i,j,k) = (0._R_P, 0._R_P)
            else
               self%f_hat(i,j,k) = self%f_hat(i,j,k) * (scale / lam)
            endif
         enddo
      enddo
   enddo
   call fftw_execute_dft_c2r(self%plan_c2r, self%f_hat, self%f_global)
#else
   if (.false.) scale = factor + lam + i + j + k
   call elisha_set_error(stat, errmsg, ELISHA_ERR_NO_FFT, 'elisha_spectral_solver_object%transform: built without '// &
                         'an FFT library')
#endif
   endsubroutine transform

   subroutine transform_kernel(self, stat, errmsg)
   !< Rank 0, isolated: `khat <- FFT(f_global) / prod(m)`, real (the kernel is even in every direction).
   class(elisha_spectral_solver_object), intent(inout)         :: self   !< The solver.
   integer(I4P),                         intent(out)           :: stat   !< Status.
   character(:), allocatable,            intent(out), optional :: errmsg !< Message.

#ifdef _FFTW
   stat = ELISHA_OK
   call fftw_execute_dft_r2c(self%plan_r2c, self%f_global, self%f_hat)
   self%khat = real(self%f_hat, R_P) / real(product(self%m), R_P)
#else
   call elisha_set_error(stat, errmsg, ELISHA_ERR_NO_FFT, 'elisha_spectral_solver_object%transform_kernel: built '// &
                         'without an FFT library')
#endif
   endsubroutine transform_kernel

   pure function root_cells_max(growth) result(n)
   !< The most cells of a multigrid root on this backend (#84 E5c, E6b), for a root whose transform (the FFT solver's
   !< extension of its bounded axes, 2 or 4 times per axis) has `growth` cells per cell: on the host a transform of at
   !< most 128^3 cells, where the distributed transform (E5d) of a large root beats smoothing more levels (medooza CPU,
   !< 3.9 M AMR cells, np 1 / 4 / 16): periodic 12.4 / 3.9 / 2.5 s with a 32^3 root, 8.6 / 2.7 / 1.6 s with 128^3;
   !< isolated (the root ghost-Dirichlet, its transform 8 times its cells)
   !< 30.4 / 9.6 / 6.3 s with 32^3, 26.8 / 8.5 / 5.2 s with 64^3, 49.2 / 15.0 / 6.2 s with 128^3. cuFFT overrides it.
   integer(I4P), intent(in) :: growth !< Transform cells per root cell.
   integer(I4P)             :: n      !< Cells.

   n = 2097152_I4P / max(growth, 1_I4P)
   endfunction root_cells_max

   subroutine slab_make_plans(self, stat, errmsg)
   !< The slab plans (#84 E5d): one 2-D plane (r2c, c2r) and one row of z-lines (h1 lines of m(3) cells, c2c), each
   !< made once on its own buffer and executed on every plane and row in turn (FFTW_ESTIMATE: deterministic, and the
   !< arithmetic of a value does not depend on how many planes or rows a rank holds).
   class(elisha_spectral_solver_object), intent(inout)         :: self   !< The solver.
   integer(I4P),                         intent(out)           :: stat   !< Status.
   character(:), allocatable,            intent(out), optional :: errmsg !< Message.
   integer(c_int)                           :: nl(1) !< Line length.

   allocate(self%w_r(self%m(1),self%m(2)), self%w_c(self%h1,self%m(2)), self%w_l(self%m(3),self%h1))
   self%w_r = 0._R_P ; self%w_c = (0._R_P, 0._R_P) ; self%w_l = (0._R_P, 0._R_P)
#ifdef _FFTW
   self%plan_pr2c = fftw_plan_dft_r2c_2d(int(self%m(2), c_int), int(self%m(1), c_int), self%w_r, self%w_c, &
                                         FFTW_ESTIMATE)
   self%plan_pc2r = fftw_plan_dft_c2r_2d(int(self%m(2), c_int), int(self%m(1), c_int), self%w_c, self%w_r, &
                                         FFTW_ESTIMATE)
   nl = int(self%m(3), c_int)
   self%plan_lfwd = fftw_plan_many_dft(1_c_int, nl, int(self%h1, c_int), self%w_l, c_null_ptr, 1_c_int, nl(1), &
                                       self%w_l, c_null_ptr, 1_c_int, nl(1), FFTW_FORWARD, FFTW_ESTIMATE)
   self%plan_lbwd = fftw_plan_many_dft(1_c_int, nl, int(self%h1, c_int), self%w_l, c_null_ptr, 1_c_int, nl(1), &
                                       self%w_l, c_null_ptr, 1_c_int, nl(1), FFTW_BACKWARD, FFTW_ESTIMATE)
   stat = ELISHA_OK
   if (.not.(c_associated(self%plan_pr2c) .and. c_associated(self%plan_pc2r) .and. c_associated(self%plan_lfwd) &
       .and. c_associated(self%plan_lbwd))) call elisha_set_error(stat, errmsg, ELISHA_ERR_FFT_PLAN, &
       'elisha_spectral_solver_object%slab_make_plans: FFTW could not make the plans')
#else
   nl = 0_c_int
   call elisha_set_error(stat, errmsg, ELISHA_ERR_NO_FFT, 'elisha_spectral_solver_object%slab_make_plans: built '// &
                         'without an FFT library (build with -D_FFTW and link fftw3)')
#endif
   endsubroutine slab_make_plans

   subroutine slab_destroy_plans(self)
   !< Destroy the slab plans.
   class(elisha_spectral_solver_object), intent(inout) :: self !< The solver.

#ifdef _FFTW
   if (c_associated(self%plan_pr2c)) call fftw_destroy_plan(self%plan_pr2c)
   if (c_associated(self%plan_pc2r)) call fftw_destroy_plan(self%plan_pc2r)
   if (c_associated(self%plan_lfwd)) call fftw_destroy_plan(self%plan_lfwd)
   if (c_associated(self%plan_lbwd)) call fftw_destroy_plan(self%plan_lbwd)
#endif
   self%plan_pr2c = c_null_ptr ; self%plan_pc2r = c_null_ptr
   self%plan_lfwd = c_null_ptr ; self%plan_lbwd = c_null_ptr
   endsubroutine slab_destroy_plans

   subroutine slab_planes(self, forward, stat, errmsg)
   !< The 2-D transform of every local plane: forward slab_r -> slab_c (r2c), backward slab_c -> slab_r (c2r), one plane
   !< at a time through the plane buffers.
   class(elisha_spectral_solver_object), intent(inout) :: self    !< The solver.
   logical,                   intent(in)    :: forward !< Direction.
   integer(I4P),                         intent(out)           :: stat   !< Status.
   character(:), allocatable,            intent(out), optional :: errmsg !< Message.
   integer(I4P)                             :: k       !< Counter.

#ifdef _FFTW
   stat = ELISHA_OK
   do k=1, self%zr(self%comm%rank+1) - self%zr(self%comm%rank)
      if (forward) then
         self%w_r = self%slab_r(:,:,k)
         call fftw_execute_dft_r2c(self%plan_pr2c, self%w_r, self%w_c)
         self%slab_c(:,:,k) = self%w_c
      else
         self%w_c = self%slab_c(:,:,k)
         call fftw_execute_dft_c2r(self%plan_pc2r, self%w_c, self%w_r)
         self%slab_r(:,:,k) = self%w_r
      endif
   enddo
#else
   k = merge(1, 0, forward)
   call elisha_set_error(stat, errmsg, ELISHA_ERR_NO_FFT, 'elisha_spectral_solver_object%slab_planes: built without '// &
                         'an FFT library')
#endif
   endsubroutine slab_planes

   subroutine slab_lines(self, forward, stat, errmsg)
   !< The 1-D transform of every local z-line, in place in pen, one row of h1 lines at a time through the row buffer.
   class(elisha_spectral_solver_object), intent(inout) :: self    !< The solver.
   logical,                   intent(in)    :: forward !< Direction.
   integer(I4P),                         intent(out)           :: stat   !< Status.
   character(:), allocatable,            intent(out), optional :: errmsg !< Message.
   integer(I4P)                             :: j       !< Counter.

#ifdef _FFTW
   stat = ELISHA_OK
   do j=1, self%yr(self%comm%rank+1) - self%yr(self%comm%rank)
      self%w_l = self%pen(:,:,j)
      if (forward) then
         call fftw_execute_dft(self%plan_lfwd, self%w_l, self%w_l)
      else
         call fftw_execute_dft(self%plan_lbwd, self%w_l, self%w_l)
      endif
      self%pen(:,:,j) = self%w_l
   enddo
#else
   j = merge(1, 0, forward)
   call elisha_set_error(stat, errmsg, ELISHA_ERR_NO_FFT, 'elisha_spectral_solver_object%slab_lines: built without '// &
                         'an FFT library')
#endif
   endsubroutine slab_lines

   subroutine slab_transpose(self, forward, stat, errmsg)
   !< forward: slab_c(x, y, z local) -> pen(z, x, y local); backward the reverse (`alltoallv_c`). This rank's own block
   !< is copied directly.
   class(elisha_spectral_solver_object), intent(inout) :: self    !< The solver.
   logical,                   intent(in)    :: forward !< Direction.
   integer(I4P),                         intent(out)           :: stat   !< Status.
   character(:), allocatable,            intent(out), optional :: errmsg !< Message.
   integer(I4P), allocatable :: sc(:), sd(:), rc(:), rd(:) !< Counts and displacements (values).
   complex(R_P), allocatable :: sb(:), rb(:)               !< Buffers.
   integer(I4P)              :: nr, me, r, c, x, y, z, nzl, nyl, z0, y0 !< Counters, sizes, first plane and row.
   type(elisha_comm_object)  :: comm                       !< Copy of self%comm (below).

   ! a field-wise copy of self%comm (not `comm = self%comm`, not a helper): passing any part of self to a procedure
   ! (self%comm%alltoallv_c) makes nvfortran 26.9 -fast leave this routine's copy loops unvectorised and unrolled less
   ! (-Minfo: 5 loops optimised against ADAM's 10, which passed MPI_COMM_WORLD), and the forward transpose of the
   ! 128^3 isolated solve took 34 ms against ADAM's 21 ms on one rank. gfortran's transpose times equal ADAM's with
   ! the copy; in solve_slabs, whose internal procedures would host-associate it, the same copy slowed gfortran's
   ! redistribution by 40%, and nvfortran's is not slower there without it: solve_slabs keeps self%comm.
   comm%comm = self%comm%comm ; comm%rank = self%comm%rank ; comm%nranks = self%comm%nranks
   comm%is_initialized = self%comm%is_initialized
   nr = comm%nranks ; me = comm%rank
   nzl = self%zr(me+1) - self%zr(me) ; nyl = self%yr(me+1) - self%yr(me)
   z0 = self%zr(me) ; y0 = self%yr(me)
   allocate(sc(0:nr-1), sd(0:nr-1), rc(0:nr-1), rd(0:nr-1))
   do r=0, nr-1
      if (forward) then
         sc(r) = self%h1 * (self%yr(r+1) - self%yr(r)) * nzl
         rc(r) = self%h1 * nyl * (self%zr(r+1) - self%zr(r))
      else
         sc(r) = self%h1 * nyl * (self%zr(r+1) - self%zr(r))
         rc(r) = self%h1 * (self%yr(r+1) - self%yr(r)) * nzl
      endif
   enddo
   sc(me) = 0 ; rc(me) = 0
   sd(0) = 0 ; rd(0) = 0
   do r=1, nr-1
      sd(r) = sd(r-1) + sc(r-1) ; rd(r) = rd(r-1) + rc(r-1)
   enddo
   allocate(sb(max(sum(sc),1)), rb(max(sum(rc),1)))
   c = 0
   do r=0, nr-1
      if (r == me) cycle
      if (forward) then
         do z=1, nzl
            do y=self%yr(r)+1, self%yr(r+1)
               do x=1, self%h1
                  c = c + 1 ; sb(c) = self%slab_c(x,y,z)
               enddo
            enddo
         enddo
      else
         do z=self%zr(r)+1, self%zr(r+1)
            do y=1, nyl
               do x=1, self%h1
                  c = c + 1 ; sb(c) = self%pen(z,x,y)
               enddo
            enddo
         enddo
      endif
   enddo
   stat = ELISHA_OK
   if (nr > 1) then
      call comm%alltoallv_c(sb, sc, sd, rb, rc, rd, stat)
      if (collective_failed(stat, errmsg, 'elisha_spectral_solver_object%slab_transpose')) return
   endif
   ! this rank's own block
   do y=1, nyl
      do x=1, self%h1
         do z=1, nzl
            if (forward) then
               self%pen(z0+z,x,y) = self%slab_c(x,y0+y,z)
            else
               self%slab_c(x,y0+y,z) = self%pen(z0+z,x,y)
            endif
         enddo
      enddo
   enddo
   c = 0
   do r=0, nr-1
      if (r == me) cycle
      if (forward) then
         do z=self%zr(r)+1, self%zr(r+1)
            do y=1, nyl
               do x=1, self%h1
                  c = c + 1 ; self%pen(z,x,y) = rb(c)
               enddo
            enddo
         enddo
      else
         do z=1, nzl
            do y=self%yr(r)+1, self%yr(r+1)
               do x=1, self%h1
                  c = c + 1 ; self%slab_c(x,y,z) = rb(c)
               enddo
            enddo
         enddo
      endif
   enddo
   endsubroutine slab_transpose

   subroutine slab_spectral(self, factor)
   !< The spectral solve on this rank's z-lines: the division by the eigenvalues (the zero modes dropped) or, isolated,
   !< the product with the kernel's spectrum.
   class(elisha_spectral_solver_object), intent(inout) :: self    !< The solver.
   real(R_P),                 intent(in)    :: factor  !< Source factor.
   real(R_P)                                :: scale   !< Factor over the unnormalised transform pair.
   real(R_P)                                :: lam     !< Eigenvalue of the mode.
   integer(I4P)                             :: i, j, k, nyl, y0 !< Counters, local rows, first row.

   nyl = self%yr(self%comm%rank+1) - self%yr(self%comm%rank) ; y0 = self%yr(self%comm%rank)
   if (self%isolated) then
      do j=1, nyl
         do i=1, self%h1
            do k=1, self%m(3)
               self%pen(k,i,j) = self%pen(k,i,j) * (factor * self%khat_pen(k,i,j))
            enddo
         enddo
      enddo
   else
      scale = factor / real(product(int(self%nt, I8P)), R_P)
      do j=1, nyl
         do i=1, self%h1
            do k=1, self%m(3)
               lam = self%lam1(i-1) + self%lam2(y0+j-1) + self%lam3(k-1)
               if (lam == 0._R_P) then
                  self%pen(k,i,j) = (0._R_P, 0._R_P)
               else
                  self%pen(k,i,j) = self%pen(k,i,j) * (scale / lam)
               endif
            enddo
         enddo
      enddo
   endif
   endsubroutine slab_spectral

   subroutine slab_keep_kernel(self)
   !< Isolated, at setup: `khat_pen <- real(pen) / prod(m)`, the kernel's spectrum (real: the kernel is even).
   class(elisha_spectral_solver_object), intent(inout) :: self !< The solver.

   self%khat_pen = real(self%pen, R_P) / real(product(int(self%m, I8P)), R_P)
   endsubroutine slab_keep_kernel

   subroutine slab_update(self, to_device)
   !< Move `slab_r` to (`to_device`) or from where the stages run: nothing on the host.
   class(elisha_spectral_solver_object), intent(inout) :: self      !< The solver.
   logical,                   intent(in)    :: to_device !< Direction.

   if (to_device .or. .not.self%slab) return
   endsubroutine slab_update

   ! module helpers
   subroutine agree_status(comm, stat, errmsg, where)
   !< Every rank takes the largest status over the ranks (`max_i4`; the codes are positive): a rank that failed keeps
   !< its message, the others get the code with "on another rank". Collective.
   type(elisha_comm_object),  intent(in)              :: comm   !< Communication context.
   integer(I4P),              intent(inout)           :: stat   !< Status.
   character(:), allocatable, intent(inout), optional :: errmsg !< Message.
   character(*),              intent(in)              :: where  !< The procedure, for the message.
   integer(I4P)                                       :: c(1)   !< Agreed code.
   integer(I4P)                                       :: st     !< Status of the reduction.

   c = stat
   call comm%max_i4(c, st)
   if (st /= ELISHA_OK) then
      call elisha_set_error(stat, errmsg, st, where//': a collective failed')
   elseif (c(1) /= stat) then
      call elisha_set_error(stat, errmsg, c(1), where//': on another rank')
   endif
   endsubroutine agree_status

   function collective_failed(stat, errmsg, where) result(failed)
   !< A collective returned a failure: the status (every rank saw it) with the procedure's message.
   integer(I4P),              intent(inout)           :: stat   !< Status of the collective.
   character(:), allocatable, intent(inout), optional :: errmsg !< Message.
   character(*),              intent(in)              :: where  !< The procedure, for the message.
   logical                                            :: failed !< The collective failed.
   integer(I4P)                                       :: code   !< Copy of the status.

   failed = stat /= ELISHA_OK
   if (.not.failed) return
   code = stat
   call elisha_set_error(stat, errmsg, code, where//': a collective failed')
   endfunction collective_failed
endmodule elisha_spectral_solver
