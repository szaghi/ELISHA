!< ELISHA verification SV-9, SV-12, SV-13, E1b, E1c, E5c: the hierarchical solver on AMR (ADAM's test_poisson_mg).
program elisha_sv_mg_mpi
!< ELISHA verification SV-9, SV-12, SV-13, E1b, E1c, E5c: the hierarchical solver on AMR (ADAM's test_poisson_mg).
!<
!< **Provenance.** ADAM's `src/tests/poisson/test_poisson_mg.F90` (ADAM issues #81 P5a, #87 P1-P2, #84 E1b, E1c, E5c),
!< ported onto ELISHA: the legs, the sources, the exact and manufactured solutions, the measures, the bounds and the
!< printed lines are ADAM's; the solvers are `elisha_hierarchical_solver_object` (ADAM's `poisson_mg_object`),
!< `elisha_spectral_solver_object` (ADAM's `poisson_fft_object`, the reference on uniform grids) and the facade
!< `elisha_solver_object` (ADAM's `flail_object`, the E1c leg); the grids are `elisha_mesh_object`s built here (ADAM
!< refined its realm's tree), and a refusal is the status code ELISHA returns (`elisha_status_name`), not a stop.
!<
!< **The grids.** ADAM's: one root block of 8^3 cells (8^2 in 2-D, null z), 3 ghost layers, on SV-2's box
!< [0,1] x [0,2] x [0,3] or SV-6's cube [-1,1]^3, uniform at a base level; the static AMR refines once more the blocks
!< whose centre lies in the central box [L/4, 3L/4] (SV-9b, SV-12b, E5c), in the corner octant [0, L/2] (E1b, E1c: leaves
!< of both levels on the low faces), in the star's box [-1/2, 1/2]^3 (SV-13b) or in the octant x, y, z > 0 (the
!< boundary-refined refusal). The leaves are numbered in Morton order and dealt in contiguous chunks to the ranks.
!<
!< **Legs** (each asserts ADAM's bounds and prints ADAM's lines, which ADAM's figure scripts parse):
!<   - `--sv9`: SV-9a (uniform: the multigrid against the spectral solver, `TOL_FFT`, at most one FCG iteration),
!<     SV-9b (the central box refined over base levels 2 and 3: the FCG residual, the symmetry defect of the assembled
!<     rows `TOL_SYMMETRY`, every face ghost filled, the order of Phi `MIN_ORDER`, the truncation growth
!<     `MAX_TRUNCATION_GROWTH`; reported: the iterations, the mean convergence factor, the 2:1 face gravity and its
!<     order), SV-12a/b (Dirichlet: the exact potential as the ghost values, `A Phi = factor V (rho - rho0) - B g`;
!<     uniform at most one iteration), and the refusals `--mixed` (`ELISHA_ERR_BC_MISSING`), `--refuse-bc-mismatch`
!<     (`ELISHA_ERR_BC_PERIODICITY`), `--refuse-bc-mixed-ghost` (`ELISHA_ERR_BC_GHOST_PAIR`);
!<   - `--isolated`: SV-13a (uniform 16^3, 32^3: the James step against the spectral solver's isolated mode,
!<     `TOL_FFT_ISOLATED`, one + one iterations), SV-13b (the star's box refined over base levels 2 and 3, the polytrope
!<     and the uniform sphere: the order of Phi), the symmetry of the solution map in situ on the first grid
!<     (`TOL_SYMMETRY_ISOLATED`), the boundary ghosts against the exact potential (reported), and the refusal
!<     `--refuse-boundary-refined` (`ELISHA_ERR_ISOLATED_BOUNDARY_LEAF`);
!<   - `--bc`: E1b, face kinds per axis (P/P D/N N/D; D/D N/N G/G; P/P N/N N/N singular; 2-D N/D P/P): uniform against
!<     the spectral solver with the same kinds; the corner refined over base levels 2 and 3 against the manufactured
!<     potential `prod_d cos(2 pi x_d / L_d + theta_d)`, its boundary values (`bnd_kind`, `bnd_dir`, `bnd_x`) the ghost
!<     values: the orders of Phi and of the domain-face ghosts, the truncation growth;
!<   - `--sub`: E5c, the levels inside the blocks forced (`root_cells = 1`) against none (`root_cells = huge`): the
!<     levels created, both residuals, `TOL_SUB`, the iterations at most `MAX_SUB_RATIO` times the exact ones (or
!<     `MAX_SUB_UNIFORM`);
!<   - `--flail`: E1c, the facade against the solver it chooses called directly, bit for bit on every cell and ghost
!<     (the spectral path on a uniform grid, the hierarchical on AMR with the manufactured boundary values given through
!<     the facade's `boundary_faces`; the wide D(G) p = 4), a vector of three facades with per-component kinds solved
!<     interleaved against three alone, and the refusals `--refuse-flail-ghost-fft` (`ELISHA_ERR_GHOST_VALUES`),
!<     `--refuse-flail-isolated-bc` (`ELISHA_ERR_ISOLATED_BC`), `--refuse-flail-wide-amr` (`ELISHA_ERR_DG_AMR`).
!< No leg flag runs every leg (the CI run). A refusal flag alone runs only that refusal: exit 0 when the setup (or the
!< solve) returns the expected code, printed by name; 1 otherwise. ADAM's run had to stop (exit 1); ELISHA returns a status.
!<
!< `--dump DIR` writes rank 0's potential of every grid the legs measure to `DIR/phi-mg-<leg>-<case>-l<base level>.bin`
!< in ADAM's dump format (3 int32 sizes, then float64 values): sizes [cells per block, leaves, 1], the leaves' interior
!< cells (i fastest) in the Morton order of the leaves, whatever the rank count: `poisson_oracle.py compare` reads them
!< (two builds or two rank counts agree to round-off; the reductions sum in another order).
!<
!< Usage: `mpirun -np N exe/elisha_sv_mg_mpi [--sv9] [--isolated] [--bc] [--sub] [--flail] [--dump DIR]`
!<        `mpirun -np N exe/elisha_sv_mg_mpi --mixed | --refuse-bc-mismatch | --refuse-bc-mixed-ghost |
!<         --refuse-boundary-refined | --refuse-flail-ghost-fft | --refuse-flail-isolated-bc | --refuse-flail-wide-amr`
!< `--device` (only in the device build, `tests-nvf-oac`, `exe-oac/`): the OpenACC hierarchical backend (its root on
!< cuFFT), the cuFFT spectral one and the facade with both molds, each rank on the GPU of its node-local rank; the same
!< bounds (the device sums its dots in another order: round-off differences, iterations within one).
!< Built with `-DELISHA_NO_MPI` it runs on one rank. The FP traps of a debug build are off around `MPI_Init` only.
#ifndef ELISHA_NO_MPI
use mpi
#endif
#ifdef _CUFFT
! tests-nvf-oac: the device backends, selected at run time by --device
use openacc
use elisha_device, only : elisha_spectral_solver_cufft_object, elisha_hierarchical_solver_oac_object
#endif
use, intrinsic :: ieee_exceptions, only : ieee_usual, ieee_get_halting_mode, ieee_set_halting_mode
use elisha, only : I4P, I8P, R_P, ELISHA_OK, ELISHA_BC_PERIODIC, ELISHA_BC_DIRICHLET, ELISHA_BC_NEUMANN,                &
                   ELISHA_BC_DIRICHLET_GHOST, ELISHA_ERR_BC_MISSING, ELISHA_ERR_BC_PERIODICITY,                       &
                   ELISHA_ERR_BC_GHOST_PAIR, ELISHA_ERR_ISOLATED_BOUNDARY_LEAF, ELISHA_ERR_GHOST_VALUES,               &
                   ELISHA_ERR_ISOLATED_BC, ELISHA_ERR_DG_AMR, elisha_status_name, elisha_comm_object,                 &
                   elisha_mesh_object, elisha_spectral_solver_object, elisha_hierarchical_solver_object,              &
                   elisha_solver_object
implicit none
real(R_P),    parameter :: PI=acos(-1._R_P)              !< Pi.
real(R_P),    parameter :: FACTOR=4._R_P * PI            !< Source factor, 4 pi G with G = 1.
real(R_P),    parameter :: RHO0=1.5_R_P                  !< Mean density.
real(R_P),    parameter :: A1=1._R_P                     !< Amplitude of the first mode.
real(R_P),    parameter :: A2=0.5_R_P                    !< Amplitude of the second mode.
real(R_P),    parameter :: L(3)=[1._R_P, 2._R_P, 3._R_P] !< Domain lengths.
real(R_P),    parameter :: SENTINEL=-1.e30_R_P           !< Unfilled phi.
real(R_P),    parameter :: TOL_FFT=1.e-8_R_P             !< Bound on |Phi_mg - Phi_fft| / max |Phi_fft|.
real(R_P),    parameter :: TOL_SYMMETRY=1.e-12_R_P       !< Bound on the operator's symmetry defect.
real(R_P),    parameter :: MIN_ORDER=1.7_R_P             !< Bound on the order of Phi against the exact one.
integer(I4P), parameter :: MAX_ITERATIONS=60_I4P         !< FCG iterations allowed.
real(R_P),    parameter :: MAX_TRUNCATION_GROWTH=1.5_R_P !< Bound on max truncation (fine grid) / (coarse grid).
real(R_P),    parameter :: A_SPHERE=0.5_R_P              !< Radius of the isolated stars (SV-13).
integer(I4P), parameter :: SUB=8_I4P                     !< Samples per direction of a cell average (uniform sphere).
integer(I4P), parameter :: PROFILE_POLYTROPE=1_I4P       !< The n = 1 polytrope.
integer(I4P), parameter :: PROFILE_UNIFORM=2_I4P         !< The uniform sphere.
real(R_P),    parameter :: TOL_ISOLATED=1.e-12_R_P       !< FCG tolerance of the isolated solves.
real(R_P),    parameter :: TOL_FFT_ISOLATED=1.e-9_R_P    !< Bound on |Phi_mg - Phi_fft| / max |Phi_fft|, isolated.
real(R_P),    parameter :: TOL_SYMMETRY_ISOLATED=1.e-9_R_P !< Bound on the solution map's symmetry defect, isolated.
real(R_P),    parameter :: MIN_ORDER_SPHERE=1.2_R_P      !< Bound on the uniform sphere's order.
real(R_P),    parameter :: TOL_SUB=1.e-7_R_P             !< Bound on |Phi_sub - Phi_exact| / max |Phi_exact| (E5c).
real(R_P),    parameter :: MAX_SUB_RATIO=1.5_R_P         !< Bound on the iterations with the levels inside the blocks over
                                                         !< those without (E5c).
integer(I4P), parameter :: MAX_SUB_UNIFORM=20_I4P        !< The same bound where the exact coarsest needs 1 iteration.
integer(I4P), parameter :: P_=ELISHA_BC_PERIODIC         !< Periodic.
integer(I4P), parameter :: D_=ELISHA_BC_DIRICHLET        !< Dirichlet at the face.
integer(I4P), parameter :: N_=ELISHA_BC_NEUMANN          !< Neumann at the face.
integer(I4P), parameter :: G_=ELISHA_BC_DIRICHLET_GHOST  !< Dirichlet at the ghost centre.
integer(I4P), parameter :: BC_A(2,3)=reshape([P_, P_, D_, N_, N_, D_], [2, 3]) !< Mixed periodicity, mixed axes.
integer(I4P), parameter :: BC_B(2,3)=reshape([D_, D_, N_, N_, G_, G_], [2, 3]) !< Every bounded kind, one per axis.
integer(I4P), parameter :: BC_C(2,3)=reshape([P_, P_, N_, N_, N_, N_], [2, 3]) !< Singular: periodic and Neumann.
integer(I4P), parameter :: BC_D(2,3)=reshape([N_, D_, P_, P_, P_, P_], [2, 3]) !< 2-D: N/D along x, periodic y.
integer(I4P), parameter :: BC_PEC_XY(2,3)=reshape([P_, P_, P_, P_, D_, D_], [2, 3]) !< PEC z faces, tangential comps.
real(R_P),    parameter :: THETA(3)=[0.3_R_P, 0.7_R_P, 1.1_R_P] !< Phases of the manufactured potential.
integer(I4P), parameter :: NB_SIDE=8_I4P                 !< Cells per block side.
integer(I4P), parameter :: NGC=3_I4P                     !< Ghost layers.
integer(I4P), parameter :: REF_NONE=0_I4P                !< Uniform.
integer(I4P), parameter :: REF_BOX=1_I4P                 !< The central box [L/4, 3L/4] refined once.
integer(I4P), parameter :: REF_CORNER=2_I4P              !< The corner octant [0, L/2] refined once.
integer(I4P), parameter :: REF_STAR=3_I4P                !< The star's box [-1/2, 1/2]^3 refined once (cube).
integer(I4P), parameter :: REF_OCTANT=4_I4P              !< The octant x, y, z > 0 refined once (cube).
integer(I4P), parameter :: SRC_RHO=0_I4P                 !< Source: SV-2's density.
integer(I4P), parameter :: SRC_RHO_DEV=1_I4P             !< Source: SV-2's density less rho0.
integer(I4P), parameter :: SRC_STAR=2_I4P                !< Source: SV-6's star (the profile given).
integer(I4P), parameter :: SRC_LAPM=3_I4P                !< Source: the manufactured Laplacian over the factor.
integer(I4P), parameter :: MAX_ROWS=64_I4P               !< Rows of the summary table.

type :: grid_t
   !< One grid of a leg: the mesh, every leaf in Morton order, this rank's share.
   type(elisha_mesh_object)  :: mesh                     !< Mesh.
   integer(I4P)              :: nk=8_I4P                 !< Cells per block along z (1: null z).
   integer(I4P)              :: nb=0_I4P                 !< This rank's blocks.
   integer(I4P)              :: nleaf=0_I4P              !< Leaves over the ranks.
   integer(I4P)              :: level=0_I4P              !< Base level.
   logical                   :: cube=.false.             !< The cube [-1,1]^3 (else the box [0,1] x [0,2] x [0,3]).
   logical                   :: null(3)=.false.          !< Null axes.
   logical                   :: active(3)=.true.         !< Active axes.
   real(R_P)                 :: lb(3)=0._R_P             !< Domain lengths.
   real(R_P)                 :: emin(3)=0._R_P           !< Domain lower corner.
   integer(I4P), allocatable :: leaves(:,:)              !< Every leaf (i, j, k, level), Morton order [4,nleaf].
   integer(I4P), allocatable :: gid(:)                   !< This rank's blocks: their leaf ids [nb].
endtype grid_t

type :: mg_box
   !< One hierarchical solver of the run's backend. check_sub keeps two in an array of this box, not in a polymorphic
   !< array `class(elisha_hierarchical_solver_object), allocatable :: mg(:)`: nvfortran 26.9 corrupts the automatic (and
   !< the explicit) deallocation of such an array when its dynamic type adds components (the OpenACC solver's), a
   !< segfault in pgf90_dealloc_mbr03a_i8 (medooza job 1369178, --sub --device; reproduced on the host with a one-logical
   !< extension; gfortran is clean). The library has no polymorphic arrays.
   class(elisha_hierarchical_solver_object), allocatable :: s !< The solver.
endtype mg_box

type(elisha_comm_object)  :: c                           !< Communication context (the world).
logical                   :: test_passed                 !< Aggregate pass flag.
logical                   :: leg(5)                      !< Legs to run: sv9, isolated, bc, sub, flail.
character(64)             :: refusal                     !< The only refusal to run ('': none).
character(256)            :: arg                         !< Command argument.
character(256)            :: dump_dir                    !< Directory of the potential dumps ('': no dump).
character(320)            :: rows(MAX_ROWS)              !< Summary rows.
integer(I4P)              :: nrows                       !< Summary rows used.
integer(I4P)              :: a, stat                     !< Counter, status.
integer                   :: ierr, world                 !< MPI error status, the world communicator.
logical                   :: halting(3)                  !< The FP halting modes of the build (debug: on).
logical                   :: device                      !< --device: the device backends.
logical                   :: per3(3)                     !< build's periodic axes, a named actual argument (ifx's
                                                         !< -check arg_temp_created reports every expression passed).
real(R_P)                 :: t0, tl                      !< Wall clock at the start, at a leg's start.
character(:), allocatable :: msg                         !< Message.

leg = .false. ; refusal = '' ; dump_dir = '' ; nrows = 0
device = .false.
a = 0
do while (a < command_argument_count())
   a = a + 1
   call get_command_argument(a, arg)
   select case(trim(arg))
   case('--sv9')      ; leg(1) = .true.
   case('--isolated') ; leg(2) = .true.
   case('--bc')       ; leg(3) = .true.
   case('--sub')      ; leg(4) = .true.
   case('--flail')    ; leg(5) = .true.
   case('--mixed')    ; refusal = 'mixed'
   case('--dump')
      a = a + 1
      call get_command_argument(a, dump_dir)
   case('--device')   ; device = .true.
   case default
      if (index(arg, '--refuse-') == 1) then
         refusal = arg(10:)
      else
         print '(A)', 'elisha_sv_mg_mpi: unknown argument "'//trim(arg)//'"'
         error stop 2
      endif
   endselect
enddo
if (.not.any(leg)) leg = .true.
#ifndef ELISHA_NO_MPI
call ieee_get_halting_mode(ieee_usual, halting)
call ieee_set_halting_mode(ieee_usual, .false.)
call MPI_INIT(ierr)
call ieee_set_halting_mode(ieee_usual, halting)
world = MPI_COMM_WORLD
#else
halting = .false. ; ierr = 0 ; world = 0
#endif
#ifdef _CUFFT
if (device) call select_device
#else
if (device) then
   print '(A)', 'elisha_sv_mg_mpi: --device needs the device build (tests-nvf-oac)'
   error stop 2
endif
#endif
call c%init(world, stat, msg)
if (stat /= ELISHA_OK) then
   print '(A)', 'elisha_sv_mg_mpi: comm init: '//msg
   error stop 1
endif
t0 = c%wtime()
test_passed = .true.
if (len_trim(refusal) > 0) then
   select case(trim(refusal))
   case('mixed', 'bc-mismatch', 'bc-mixed-ghost', 'boundary-refined', 'flail-ghost-fft', 'flail-isolated-bc', &
        'flail-wide-amr')
      call check_refusal(trim(refusal))
   case default
      if (c%rank == 0) print '(A)', 'elisha_sv_mg_mpi: unknown refusal "'//trim(refusal)//'"'
      call finish
      error stop 2
   endselect
else
   if (leg(1)) then
      tl = c%wtime()
      if (c%rank == 0) print '(A)', 'SV-9, SV-12: the hierarchical solver, periodic and Dirichlet'
      call check_uniform(label='SV-9a, uniform 3-D octree,  ', nk=8_I4P, level=2_I4P)
      call check_uniform(label='SV-9a, uniform 2-D quadtree,', nk=1_I4P, level=2_I4P)
      call check_amr(label='SV-9b, AMR 3-D octree,      ', nk=8_I4P, levels=[2_I4P, 3_I4P])
      call check_amr(label='SV-9b, AMR 2-D quadtree,    ', nk=1_I4P, levels=[2_I4P, 3_I4P])
      call check_dirichlet(label='SV-12a, Dirichlet, uniform 3-D octree,', nk=8_I4P, levels=[1_I4P, 2_I4P], &
                           amr=.false.)
      call check_dirichlet(label='SV-12b, Dirichlet, AMR 3-D octree,    ', nk=8_I4P, levels=[2_I4P, 3_I4P], &
                           amr=.true.)
      call check_dirichlet(label='SV-12b, Dirichlet, AMR 2-D quadtree,  ', nk=1_I4P, levels=[2_I4P, 3_I4P], &
                           amr=.true.)
      call check_refusal('mixed')
      call check_refusal('bc-mismatch')
      call check_refusal('bc-mixed-ghost')
      call leg_time('sv9')
   endif
   if (leg(2)) then
      tl = c%wtime()
      if (c%rank == 0) print '(A)', 'SV-13: the hierarchical solver, isolated (the James step)'
      call check_isolated(label='SV-13a, isolated, uniform, polytrope n=1,', prof=PROFILE_POLYTROPE, &
                          levels=[1_I4P, 2_I4P], amr=.false.)
      call check_isolated(label='SV-13b, isolated, AMR, polytrope n=1,    ', prof=PROFILE_POLYTROPE, &
                          levels=[2_I4P, 3_I4P], amr=.true.)
      call check_isolated(label='SV-13b, isolated, AMR, uniform sphere,   ', prof=PROFILE_UNIFORM, &
                          levels=[2_I4P, 3_I4P], amr=.true.)
      call check_refusal('boundary-refined')
      call leg_time('sv13')
   endif
   if (leg(3)) then
      tl = c%wtime()
      if (c%rank == 0) print '(A)', 'E1b: the hierarchical solver, boundary kinds per face'
      call check_bc_uniform(label='E1b, uniform 3-D, P/P D/N N/D,', bc=BC_A, nk=8_I4P, level=1_I4P)
      call check_bc_uniform(label='E1b, uniform 3-D, D/D N/N G/G,', bc=BC_B, nk=8_I4P, level=1_I4P)
      call check_bc_uniform(label='E1b, uniform 3-D, P/P N/N N/N,', bc=BC_C, nk=8_I4P, level=1_I4P)
      call check_bc_uniform(label='E1b, uniform 2-D, N/D P/P,    ', bc=BC_D, nk=1_I4P, level=2_I4P)
      call check_bc_amr(label='E1b, AMR 3-D, P/P D/N N/D,', bc=BC_A, nk=8_I4P, levels=[2_I4P, 3_I4P])
      call check_bc_amr(label='E1b, AMR 3-D, D/D N/N G/G,', bc=BC_B, nk=8_I4P, levels=[2_I4P, 3_I4P])
      call check_bc_amr(label='E1b, AMR 3-D, P/P N/N N/N,', bc=BC_C, nk=8_I4P, levels=[2_I4P, 3_I4P])
      call check_bc_amr(label='E1b, AMR 2-D, N/D P/P,    ', bc=BC_D, nk=1_I4P, levels=[2_I4P, 3_I4P])
      call leg_time('bc')
   endif
   if (leg(4)) then
      tl = c%wtime()
      if (c%rank == 0) print '(A)', 'E5c: the levels inside the blocks'
      call check_sub(label='E5c, uniform 3-D octree, periodic,       ', nk=8_I4P, level=2_I4P, amr=.false.)
      call check_sub(label='E5c, AMR 3-D octree, periodic,           ', nk=8_I4P, level=2_I4P, amr=.true.)
      call check_sub(label='E5c, AMR 2-D quadtree, periodic,         ', nk=1_I4P, level=2_I4P, amr=.true.)
      call check_sub(label='E5c, AMR 3-D octree, D/D N/N G/G,        ', nk=8_I4P, level=2_I4P, amr=.true., bc=BC_B)
      call check_sub(label='E5c, AMR 3-D octree, P/P D/N N/D,        ', nk=8_I4P, level=2_I4P, amr=.true., bc=BC_A)
      call check_sub(label='E5c, AMR 3-D octree, isolated (James),   ', nk=8_I4P, level=2_I4P, amr=.true., &
                     isolated=.true.)
      call leg_time('sub')
   endif
   if (leg(5)) then
      tl = c%wtime()
      if (c%rank == 0) print '(A)', 'E1c: the facade'
      call check_flail_forward(label='E1c, facade, uniform 3-D, P/P D/N N/D (FFT),      ', bc=BC_A, nk=8_I4P, &
                               level=1_I4P, amr=.false.)
      call check_flail_forward(label='E1c, facade, AMR 3-D, D/D N/N G/G (multigrid),    ', bc=BC_B, nk=8_I4P, &
                               level=2_I4P, amr=.true.)
      call check_flail_forward(label='E1c, facade, AMR 2-D, N/D P/P (multigrid),        ', bc=BC_D, nk=1_I4P, &
                               level=2_I4P, amr=.true.)
      call check_flail_forward(label='E2a, facade, uniform 3-D, P/P D/N N/D, D(G) p = 4,  ', bc=BC_A, nk=8_I4P, &
                               level=1_I4P, amr=.false., dg_order=4_I4P)
      call check_flail_vector(label='E1c, facade, PEC z faces, 3 instances, uniform (FFT),', amr=.false.)
      call check_flail_vector(label='E1c, facade, PEC z faces, 3 instances, AMR (multigrid),', amr=.true.)
      call check_refusal('flail-ghost-fft')
      call check_refusal('flail-isolated-bc')
      call check_refusal('flail-wide-amr')
      call leg_time('flail')
   endif
endif
if (c%rank == 0) then
   print '(A,I0,A,F8.2,A)', 'elisha_sv_mg_mpi summary (np ', c%nranks, ', wall ', c%wtime() - t0, ' s):'
   do a=1, nrows
      print '(A)', '  '//trim(rows(a))
   enddo
   if (test_passed) then
      print '(A)', 'elisha_sv_mg_mpi: PASSED'
   else
      print '(A)', 'elisha_sv_mg_mpi: FAILED'
   endif
endif
call finish
if (.not.test_passed) error stop 1

contains
   subroutine finish
   !< Free the context, finalise MPI.
   call c%destroy(stat)
#ifndef ELISHA_NO_MPI
   call MPI_FINALIZE(ierr)
#endif
   endsubroutine finish

   subroutine add_row(row)
   !< Append a row to the summary table.
   character(*), intent(in) :: row !< Row.

   if (nrows >= MAX_ROWS) return
   nrows = nrows + 1
   rows(nrows) = row
   endsubroutine add_row

   subroutine leg_time(name)
   !< A summary row with the wall time of a leg.
   character(*), intent(in) :: name !< Leg.
   character(16)            :: w    !< Buffer.

   write(w, '(F10.2)') c%wtime() - tl
   call add_row('leg '//name//': wall '//trim(adjustl(w))//' s')
   endsubroutine leg_time

   subroutine verdict(label, ok, what)
   !< Fold a check into the verdict and the summary.
   character(*), intent(in) :: label !< Check label.
   logical,      intent(in) :: ok    !< Pass flag.
   character(*), intent(in) :: what  !< The measures, with their bounds.

   test_passed = test_passed .and. ok
   call add_row(trim(label)//' '//what//' | '//merge('PASS', 'FAIL', ok))
   endsubroutine verdict

   ! ---------------------------------------------------------------------------------------------------------------------
   ! the legs
   ! ---------------------------------------------------------------------------------------------------------------------
   subroutine check_uniform(label, nk, level)
   !< SV-9a: the multigrid on a uniform grid against the spectral solver.
   character(*), intent(in) :: label !< Label.
   integer(I4P), intent(in) :: nk    !< Cells per block along z.
   integer(I4P), intent(in) :: level !< Uniform level.
   type(grid_t)                        :: gr       !< Grid.
   class(elisha_hierarchical_solver_object), allocatable :: mg   !< Multigrid solver.
   class(elisha_spectral_solver_object), allocatable :: fft      !< Spectral solver.
   real(R_P), allocatable   :: buf(:), phi_mg(:,:,:,:), phi_fft(:,:,:,:) !< Source, fields.
   real(R_P)                :: d(2)  !< Max difference, max |Phi_fft|.
   logical                  :: ok    !< Pass flag.

   call new_mg(mg)
   call new_fft(fft)
   per3 = per_all(nk) ! named: no argument temporary
   call build(gr, nk, level, REF_NONE, .false., per3, stat, msg) ; if (failed('build')) return
   buf = source(gr, SRC_RHO)
   call alloc_phi(gr, phi_mg) ; call alloc_phi(gr, phi_fft)
   call mg%setup(gr%mesh, c, stat, msg, max_iterations=MAX_ITERATIONS) ; if (failed('mg setup')) return
   call mg%solve_packed(gr%mesh, buf, FACTOR, phi_mg, stat, msg) ; if (failed('mg solve')) return
   call fft%setup(gr%mesh, c, stat, msg) ; if (failed('fft setup')) return
   call fft%solve_packed(gr%mesh, buf, FACTOR, phi_fft, stat, msg) ; if (failed('fft solve')) return
   call interior_diff(gr, phi_mg, phi_fft, d)
   ok = d(1) / d(2) <= TOL_FFT .and. mg%iterations <= 1
   if (c%rank == 0) print '(A)', '   '//label//' |Phi_mg - Phi_fft| / max|Phi| '//fmt(d(1) / d(2))//', iterations '// &
      itoa(mg%iterations)//', residual '//fmt(mg%residual)//merge('  PASS', '  FAIL', ok)
   call dump_leaves(gr, phi_mg, 'sv9a-'//merge('2d', '3d', nk == 1))
   call verdict(label, ok, '|mg-fft|/max '//fmt(d(1) / d(2))//' (<= '//fmt(TOL_FFT)//'), iterations '// &
                itoa(mg%iterations)//' (<= 1)')
   call mg%destroy ; call fft%destroy ; call gr%mesh%destroy
   endsubroutine check_uniform

   subroutine check_amr(label, nk, levels)
   !< SV-9b: static AMR, two base levels, each with the central box refined one level more.
   character(*), intent(in) :: label     !< Label.
   integer(I4P), intent(in) :: nk        !< Cells per block along z.
   integer(I4P), intent(in) :: levels(2) !< Base levels.
   real(R_P)                :: e_phi(2), e_grav(2), h(2), trunc(2), sym(2), order, order_g !< Measures.
   integer(I4P)             :: unfilled(2), its(2), m !< Counters.
   logical                  :: ok        !< Pass flag.

   ok = .true.
   do m=1, 2
      block
      type(grid_t)                            :: gr  !< Grid (a fresh one per level).
      class(elisha_hierarchical_solver_object), allocatable :: mg  !< Solver.
      real(R_P), allocatable :: buf(:), phi(:,:,:,:) !< Source, potential.
      call new_mg(mg)
      per3 = per_all(nk) ! named: no argument temporary
      call build(gr, nk, levels(m), REF_BOX, .false., per3, stat, msg) ; if (failed('build')) return
      buf = source(gr, SRC_RHO)
      call alloc_phi(gr, phi)
      call mg%setup(gr%mesh, c, stat, msg, max_iterations=MAX_ITERATIONS) ; if (failed('mg setup')) return
      call mg%solve_packed(gr%mesh, buf, FACTOR, phi, stat, msg) ; if (failed('mg solve')) return
      sym(m) = symmetry(gr, mg)
      call truncation(gr, mg, trunc(m))
      call measure(gr, phi, e_phi(m), e_grav(m), unfilled(m), periodic=.true.)
      h(m) = gr%lb(1) / NB_SIDE / 2._R_P**levels(m)
      its(m) = mg%iterations
      ok = ok .and. mg%residual <= mg%tolerance .and. sym(m) <= TOL_SYMMETRY .and. unfilled(m) == 0
      if (c%rank == 0) print '(A)', '   '//label//' base level '//itoa(levels(m))//', leaves '//itoa(gr%nleaf)// &
         ' (levels '//itoa(levels(m))//'-'//itoa(gr%mesh%lmax)//'): iterations '//itoa(mg%iterations)//          &
         ', factor '//fmt(mg%factor_mean)//', symmetry '//fmt(sym(m))//', truncation '//fmt(trunc(m))//          &
         ', |Phi - Phi_exact| '//fmt(e_phi(m))//', 2:1 face gravity error '//fmt(e_grav(m))//', unfilled '//     &
         itoa(unfilled(m))
      call dump_leaves(gr, phi, 'sv9b-'//merge('2d', '3d', nk == 1))
      call mg%destroy ; call gr%mesh%destroy
      endblock
   enddo
   order = log(e_phi(1) / e_phi(2)) / log(h(1) / h(2))
   order_g = log(e_grav(1) / e_grav(2)) / log(h(1) / h(2))
   ok = ok .and. order >= MIN_ORDER .and. trunc(2) / trunc(1) <= MAX_TRUNCATION_GROWTH
   if (c%rank == 0) print '(A)', '   '//label//' order of Phi '//fmtf(order)//' (min '//fmtf(MIN_ORDER)//               &
      '), truncation growth '//fmtf(trunc(2) / trunc(1))//' (max '//fmtf(MAX_TRUNCATION_GROWTH)//                  &
      '), order of the 2:1 face gravity '//fmtf(order_g)//merge('  PASS', '  FAIL', ok)
   call verdict(label, ok, 'iterations '//itoa(its(1))//', '//itoa(its(2))//', symmetry '//fmt(maxval(sym))//         &
                ' (<= '//fmt(TOL_SYMMETRY)//'), order '//fmtf(order)//' (>= '//fmtf(MIN_ORDER)//'), trunc growth '// &
                fmtf(trunc(2) / trunc(1))//' (<= '//fmtf(MAX_TRUNCATION_GROWTH)//'), 2:1 gravity order '//          &
                fmtf(order_g)//', unfilled '//itoa(sum(unfilled)))
   endsubroutine check_amr

   subroutine check_dirichlet(label, nk, levels, amr)
   !< SV-12: Dirichlet boundaries, the exact potential as the ghost values; two base levels, uniform or with the box
   !< refined one level more.
   character(*), intent(in) :: label     !< Label.
   integer(I4P), intent(in) :: nk        !< Cells per block along z.
   integer(I4P), intent(in) :: levels(2) !< Base levels.
   logical,      intent(in) :: amr       !< Refine the central box.
   real(R_P)                :: e_phi(2), e_grav(2), h(2), trunc(2), sym(2), order !< Measures.
   integer(I4P)             :: unfilled(2), its(2), m, i, nbnd(1) !< Counters.
   logical                  :: ok        !< Pass flag.

   ok = .true.
   do m=1, 2
      block
      type(grid_t)                            :: gr  !< Grid.
      class(elisha_hierarchical_solver_object), allocatable :: mg  !< Solver.
      real(R_P), allocatable :: buf(:), phi(:,:,:,:), ghost(:) !< Source, potential, ghost values.
      call new_mg(mg)
      per3 = [.false., .false., .false.] ! named: no argument temporary
      call build(gr, nk, levels(m), merge(REF_BOX, REF_NONE, amr), .false., per3, stat, msg)
      if (failed('build')) return
      buf = source(gr, SRC_RHO_DEV) ! the exact potential solves lap Phi = factor (rho - rho0)
      call alloc_phi(gr, phi)
      call mg%setup(gr%mesh, c, stat, msg, max_iterations=MAX_ITERATIONS) ; if (failed('mg setup')) return
      allocate(ghost(max(mg%nbnd,1)))
      ghost = 0._R_P
      do i=1, mg%nbnd
         ghost(i) = phi_exact(mg%bnd_x(:,i), gr%active)
      enddo
      call mg%solve_packed(gr%mesh, buf, FACTOR, phi, stat, msg, ghost=ghost) ; if (failed('mg solve')) return
      sym(m) = symmetry(gr, mg)
      call truncation(gr, mg, trunc(m), ghost=ghost, source_mean=0._R_P)
      call measure(gr, phi, e_phi(m), e_grav(m), unfilled(m), periodic=.false.)
      h(m) = gr%lb(1) / NB_SIDE / 2._R_P**levels(m)
      its(m) = mg%iterations
      nbnd = mg%nbnd
      call isum(nbnd)
      ok = ok .and. mg%residual <= mg%tolerance .and. sym(m) <= TOL_SYMMETRY .and. unfilled(m) == 0
      if (.not.amr) ok = ok .and. mg%iterations <= 1
      if (c%rank == 0) print '(A)', '   '//label//' base level '//itoa(levels(m))//', boundary faces '// &
         itoa(nbnd(1))//': iterations '//itoa(mg%iterations)//', symmetry '//fmt(sym(m))//', truncation '//  &
         fmt(trunc(m))//', |Phi - Phi_exact| '//fmt(e_phi(m))//', unfilled '//itoa(unfilled(m))
      call dump_leaves(gr, phi, 'sv12'//merge('b', 'a', amr)//'-'//merge('2d', '3d', nk == 1))
      call mg%destroy ; call gr%mesh%destroy
      endblock
   enddo
   order = log(e_phi(1) / e_phi(2)) / log(h(1) / h(2))
   ok = ok .and. order >= MIN_ORDER .and. trunc(2) / trunc(1) <= MAX_TRUNCATION_GROWTH
   if (c%rank == 0) print '(A)', '   '//label//' order of Phi '//fmtf(order)//' (min '//fmtf(MIN_ORDER)// &
      '), truncation growth '//fmtf(trunc(2) / trunc(1))//' (max '//fmtf(MAX_TRUNCATION_GROWTH)//')'//   &
      merge(', iterations at most 1', '                      ', .not.amr)//merge('  PASS', '  FAIL', ok)
   call verdict(label, ok, 'iterations '//itoa(its(1))//', '//itoa(its(2))//merge(' (<= 1)', '       ', .not.amr)// &
                ', symmetry '//fmt(maxval(sym))//' (<= '//fmt(TOL_SYMMETRY)//'), order '//fmtf(order)//            &
                ' (>= '//fmtf(MIN_ORDER)//'), trunc growth '//fmtf(trunc(2) / trunc(1))//' (<= '//                 &
                fmtf(MAX_TRUNCATION_GROWTH)//'), unfilled '//itoa(sum(unfilled)))
   endsubroutine check_dirichlet

   subroutine check_isolated(label, prof, levels, amr)
   !< SV-13: isolated boundaries (the James step) on two grids, uniform (against the spectral solver) or with the star's
   !< box refined once; the order against the exact potential, and the symmetry of the solution map on the first grid.
   character(*), intent(in) :: label     !< Label.
   integer(I4P), intent(in) :: prof      !< Source.
   integer(I4P), intent(in) :: levels(2) !< Base levels.
   logical,      intent(in) :: amr       !< Refine the star's box.
   real(R_P)                :: e_phi(2), h(2), e_ghost(2), sym, order, d(2), dfft, dots(2), min_ord !< Measures.
   integer(I4P)             :: unfilled(2), its0(2), its(2), m, i, nl !< Counters.
   logical                  :: ok        !< Pass flag.
   character(:), allocatable :: note     !< The extra measures of a grid.

   ok = .true. ; sym = 0._R_P ; dfft = 0._R_P
   do m=1, 2
      block
      type(grid_t)                            :: gr  !< Grid.
      class(elisha_hierarchical_solver_object), allocatable :: mg  !< Solver.
      class(elisha_spectral_solver_object), allocatable :: fft !< Spectral solver (uniform grids).
      real(R_P), allocatable :: buf(:), phi(:,:,:,:), phi_fft(:,:,:,:), x(:), y(:), ux(:), uy(:) !< Fields, vectors.
      call new_mg(mg)
      call new_fft(fft)
      per3 = [.false., .false., .false.] ! named: no argument temporary
      call build(gr, 8_I4P, levels(m), merge(REF_STAR, REF_NONE, amr), .true., per3, stat, msg)
      if (failed('build')) return
      buf = source(gr, SRC_STAR, prof)
      call alloc_phi(gr, phi)
      call mg%setup(gr%mesh, c, stat, msg, tolerance=TOL_ISOLATED, max_iterations=MAX_ITERATIONS, isolated=.true.)
      if (failed('mg setup')) return
      call mg%solve_packed(gr%mesh, buf, FACTOR, phi, stat, msg) ; if (failed('mg solve')) return
      ok = ok .and. mg%residual <= mg%tolerance_used
      call measure_star(gr, phi, prof, e_phi(m), e_ghost(m), unfilled(m))
      ok = ok .and. unfilled(m) == 0
      h(m) = gr%lb(1) / NB_SIDE / 2._R_P**levels(m)
      its0(m) = mg%iterations0 ; its(m) = mg%iterations
      note = ', leaves '//itoa(gr%nleaf)//': iterations '//itoa(mg%iterations0)//' + '//itoa(mg%iterations)
      call dump_leaves(gr, phi, 'sv13'//merge('b', 'a', amr)//'-p'//itoa(prof))
      if (.not.amr) then ! the multigrid against the spectral solver's isolated mode
         call alloc_phi(gr, phi_fft)
         call fft%setup(gr%mesh, c, stat, msg, isolated=.true.) ; if (failed('fft setup')) return
         call fft%solve_packed(gr%mesh, buf, FACTOR, phi_fft, stat, msg) ; if (failed('fft solve')) return
         call interior_diff(gr, phi, phi_fft, d)
         dfft = max(dfft, d(1) / d(2))
         ok = ok .and. d(1) / d(2) <= TOL_FFT_ISOLATED .and. mg%iterations0 <= 1 .and. mg%iterations <= 1
         note = note//', |Phi_mg - Phi_fft| / max|Phi| '//fmt(d(1) / d(2))
         call fft%destroy
      endif
      if (m == 1) then ! the symmetry of the solution map, in situ
         nl = gr%nb * ncb(gr)
         allocate(x(max(nl,1)), y(max(nl,1)), ux(max(nl,1)), uy(max(nl,1)))
         do i=1, nl
            x(i) = sin(1.3_R_P * i + 0.7_R_P * c%rank) ; y(i) = cos(2.1_R_P * i - 0.4_R_P * c%rank)
         enddo
         call mg%solve_packed(gr%mesh, x, 1._R_P, phi, stat, msg) ; if (failed('mg solve x')) return
         ux = mg%u(1:max(nl,1))
         call mg%solve_packed(gr%mesh, y, 1._R_P, phi, stat, msg) ; if (failed('mg solve y')) return
         uy = mg%u(1:max(nl,1))
         dots = 0._R_P
         do i=1, nl
            dots(1) = dots(1) + mg%vol(i) * x(i) * uy(i)
            dots(2) = dots(2) + mg%vol(i) * y(i) * ux(i)
         enddo
         call c%sum_r(dots, stat)
         sym = abs(dots(1) - dots(2)) / abs(dots(1))
         ok = ok .and. sym <= TOL_SYMMETRY_ISOLATED
         note = note//', symmetry of V K '//fmt(sym)
      endif
      if (c%rank == 0) print '(A)', '   '//label//' base level '//itoa(levels(m))//note//', |Phi - Phi_exact| '// &
         fmt(e_phi(m))//', boundary ghosts '//fmt(e_ghost(m))//', unfilled '//itoa(unfilled(m))
      call mg%destroy ; call gr%mesh%destroy
      endblock
   enddo
   order = log(e_phi(1) / e_phi(2)) / log(h(1) / h(2))
   min_ord = MIN_ORDER
   if (prof == PROFILE_UNIFORM) min_ord = MIN_ORDER_SPHERE
   ok = ok .and. order >= min_ord
   if (c%rank == 0) print '(A)', '   '//label//' order of Phi '//fmtf(order)//' (min '//fmtf(min_ord)//')'// &
      merge('  PASS', '  FAIL', ok)
   note = ''
   if (.not.amr) note = ', |mg-fft|/max '//fmt(dfft)//' (<= '//fmt(TOL_FFT_ISOLATED)//')'
   call verdict(label, ok, 'iterations '//itoa(its0(1))//'+'//itoa(its(1))//', '//itoa(its0(2))//'+'//itoa(its(2))// &
                note//', symmetry VK '//fmt(sym)//' (<= '//fmt(TOL_SYMMETRY_ISOLATED)//'), order '//fmtf(order)//   &
                ' (>= '//fmtf(min_ord)//'), ghosts '//fmt(e_ghost(1))//' -> '//fmt(e_ghost(2))//', unfilled '//     &
                itoa(sum(unfilled)))
   endsubroutine check_isolated

   subroutine check_bc_uniform(label, bc, nk, level)
   !< E1b: the multigrid with face kinds `bc` on a uniform grid against the spectral solver with the same kinds, the
   !< homogeneous problem with SV-2's source.
   character(*), intent(in) :: label   !< Label.
   integer(I4P), intent(in) :: bc(2,3) !< Kinds per face [lo/hi, axis].
   integer(I4P), intent(in) :: nk      !< Cells per block along z.
   integer(I4P), intent(in) :: level   !< Uniform level.
   type(grid_t)                            :: gr  !< Grid.
   class(elisha_hierarchical_solver_object), allocatable :: mg  !< Multigrid solver.
   class(elisha_spectral_solver_object), allocatable :: fft !< Spectral solver.
   real(R_P), allocatable   :: buf(:), phi_mg(:,:,:,:), phi_fft(:,:,:,:) !< Source, fields.
   real(R_P)                :: d(2)    !< Max difference, max |Phi_fft|.
   logical                  :: ok      !< Pass flag.

   call new_mg(mg)
   call new_fft(fft)
   per3 = per_bc(bc, nk) ! named: no argument temporary
   call build(gr, nk, level, REF_NONE, .false., per3, stat, msg) ; if (failed('build')) return
   buf = source(gr, SRC_RHO)
   call alloc_phi(gr, phi_mg) ; call alloc_phi(gr, phi_fft)
   call mg%setup(gr%mesh, c, stat, msg, max_iterations=MAX_ITERATIONS, bc=bc) ; if (failed('mg setup')) return
   call mg%solve_packed(gr%mesh, buf, FACTOR, phi_mg, stat, msg) ; if (failed('mg solve')) return
   call fft%setup(gr%mesh, c, stat, msg, bc=bc) ; if (failed('fft setup')) return
   call fft%solve_packed(gr%mesh, buf, FACTOR, phi_fft, stat, msg) ; if (failed('fft solve')) return
   call interior_diff(gr, phi_mg, phi_fft, d)
   ok = d(1) / d(2) <= TOL_FFT .and. mg%iterations <= 1 .and. &
        (mg%singular .eqv. .not.any(bc == ELISHA_BC_DIRICHLET .or. bc == ELISHA_BC_DIRICHLET_GHOST))
   if (c%rank == 0) print '(A)', '   '//label//' |Phi_mg - Phi_fft| / max|Phi| '//fmt(d(1) / d(2))//', iterations '// &
      itoa(mg%iterations)//', singular '//merge('T', 'F', mg%singular)//merge('  PASS', '  FAIL', ok)
   call verdict(label, ok, '|mg-fft|/max '//fmt(d(1) / d(2))//' (<= '//fmt(TOL_FFT)//'), iterations '// &
                itoa(mg%iterations)//' (<= 1), singular '//merge('T', 'F', mg%singular))
   call mg%destroy ; call fft%destroy ; call gr%mesh%destroy
   endsubroutine check_bc_uniform

   subroutine check_bc_amr(label, bc, nk, levels)
   !< E1b: the face kinds `bc` with the corner refined over two base levels, against the manufactured potential, its
   !< boundary values the ghost values of each kind.
   character(*), intent(in) :: label     !< Label.
   integer(I4P), intent(in) :: bc(2,3)   !< Kinds per face [lo/hi, axis].
   integer(I4P), intent(in) :: nk        !< Cells per block along z.
   integer(I4P), intent(in) :: levels(2) !< Base levels.
   real(R_P)                :: e_phi(2), h(2), trunc(2), sym(2), order, vsum(3), v, xyz(3) !< Measures.
   real(R_P)                :: e_gh(2), order_gh, pg(3), hd !< Domain-face ghost error, its order, ghost centre.
   real(R_P)                :: v3(3)     !< The three maxima, reduced.
   integer(I4P)             :: unfilled(2), its(2), m, i, j, k, b, cc, ijk(3), nb(3), dd, s !< Counters.
   integer(I4P)             :: nbnd(5)   !< Boundary faces over the ranks, then per kind.
   logical                  :: ok        !< Pass flag.

   ok = .true.
   do m=1, 2
      block
      type(grid_t)                            :: gr  !< Grid.
      class(elisha_hierarchical_solver_object), allocatable :: mg  !< Solver.
      real(R_P), allocatable :: buf(:), phi(:,:,:,:), ghost(:), pe(:), ye(:), yb(:) !< Fields, vectors.
      real(R_P)              :: cdx(3) !< Cell spacing.
      call new_mg(mg)
      per3 = per_bc(bc, nk) ! named: no argument temporary
      call build(gr, nk, levels(m), REF_CORNER, .false., per3, stat, msg) ; if (failed('build')) return
      buf = source(gr, SRC_LAPM)
      call alloc_phi(gr, phi)
      call mg%setup(gr%mesh, c, stat, msg, max_iterations=MAX_ITERATIONS, bc=bc) ; if (failed('mg setup')) return
      allocate(ghost(max(mg%nbnd,1)))
      ghost = 0._R_P
      nbnd = 0
      nbnd(1) = mg%nbnd
      do i=1, mg%nbnd
         ghost(i) = face_value(mg%bnd_kind(i), mg%bnd_dir(i), mg%bnd_x(:,i), gr%active)
         nbnd(mg%bnd_kind(i)+2) = nbnd(mg%bnd_kind(i)+2) + 1
      enddo
      call mg%solve_packed(gr%mesh, buf, FACTOR, phi, stat, msg, ghost=ghost) ; if (failed('mg solve')) return
      sym(m) = symmetry(gr, mg)
      ! the manufactured potential at the centres, its volume mean (singular: Phi is compared up to a constant)
      allocate(pe(max(gr%nb*ncb(gr),1)), ye(max(gr%nb*ncb(gr),1)), yb(max(gr%nb*ncb(gr),1)))
      vsum = 0._R_P
      cc = 0
      do b=1, gr%nb
         v = cell_volume(gr, b)
         do k=1, nk
            do j=1, NB_SIDE
               do i=1, NB_SIDE
                  cc = cc + 1
                  pe(cc) = phi_m(cell_centre(gr, b, i, j, k), gr%active)
                  vsum = vsum + [v, v * pe(cc), v * phi(i,j,k,b)]
               enddo
            enddo
         enddo
      enddo
      call c%sum_r(vsum, stat)
      vsum(2:3) = vsum(2:3) / vsum(1)
      if (.not.mg%singular) vsum(2:3) = 0._R_P
      ! the truncation error |(A Phi_m + B g) / V - lap Phi_m|, the error of Phi, the domain-face ghosts, the unfilled
      call mg%apply_a(pe, ye, stat) ; if (failed('apply_a')) return
      yb = 0._R_P
      call mg%apply_b(ghost, yb)
      trunc(m) = 0._R_P ; e_phi(m) = 0._R_P ; e_gh(m) = 0._R_P ; unfilled(m) = 0
      cc = 0
      do b=1, gr%nb
         v = cell_volume(gr, b)
         cdx = cell_dx(gr, b)
         do k=1, nk
            do j=1, NB_SIDE
               do i=1, NB_SIDE
                  cc = cc + 1
                  xyz = cell_centre(gr, b, i, j, k)
                  trunc(m) = max(trunc(m), abs((ye(cc) + yb(cc)) / v - lap_m(xyz, gr%active)))
                  e_phi(m) = max(e_phi(m), abs((phi(i,j,k,b) - vsum(3)) - (pe(cc) - vsum(2))))
                  do dd=1, 3
                     if (.not.gr%active(dd)) cycle
                     do s=-1, 1, 2
                        ijk = [i, j, k] ; nb = ijk ; nb(dd) = nb(dd) + s
                        if (nb(dd) >= 1 .and. nb(dd) <= nblk(gr, dd)) cycle
                        if (phi(nb(1),nb(2),nb(3),b) == SENTINEL) unfilled(m) = unfilled(m) + 1
                        if (all(bc(:,dd) == ELISHA_BC_PERIODIC)) cycle ! a domain face of a bounded axis only
                        hd = cdx(dd)
                        pg = xyz ; pg(dd) = pg(dd) + s * hd
                        if (pg(dd) > gr%emin(dd) .and. pg(dd) < gr%emin(dd) + gr%lb(dd)) cycle
                        e_gh(m) = max(e_gh(m), abs((phi(nb(1),nb(2),nb(3),b) - vsum(3)) - (phi_m(pg, gr%active) - &
                                      vsum(2))))
                     enddo
                  enddo
               enddo
            enddo
         enddo
      enddo
      v3 = [trunc(m), e_phi(m), e_gh(m)] ! named: no argument temporary
      call allmax(v3, trunc(m), e_phi(m), e_gh(m))
      call isum(unfilled(m:m))
      call isum(nbnd)
      h(m) = gr%lb(1) / NB_SIDE / 2._R_P**levels(m)
      its(m) = mg%iterations
      ok = ok .and. mg%residual <= mg%tolerance_used .and. sym(m) <= TOL_SYMMETRY .and. unfilled(m) == 0
      if (c%rank == 0) print '(A)', '   '//label//' base level '//itoa(levels(m))//', boundary faces '//itoa(nbnd(1))// &
         ' (D '//itoa(nbnd(ELISHA_BC_DIRICHLET+2))//', N '//itoa(nbnd(ELISHA_BC_NEUMANN+2))//', G '//               &
         itoa(nbnd(ELISHA_BC_DIRICHLET_GHOST+2))//'): iterations '//itoa(mg%iterations)//', symmetry '//fmt(sym(m))// &
         ', truncation '//fmt(trunc(m))//', |Phi - Phi_m| '//fmt(e_phi(m))//', domain-face ghosts '//fmt(e_gh(m))//  &
         ', unfilled '//itoa(unfilled(m))
      call mg%destroy ; call gr%mesh%destroy
      endblock
   enddo
   order = log(e_phi(1) / e_phi(2)) / log(h(1) / h(2))
   order_gh = log(e_gh(1) / e_gh(2)) / log(h(1) / h(2))
   ok = ok .and. order >= MIN_ORDER .and. order_gh >= MIN_ORDER .and. trunc(2) / trunc(1) <= MAX_TRUNCATION_GROWTH
   if (c%rank == 0) print '(A)', '   '//label//' order of Phi '//fmtf(order)//', of the domain-face ghosts '// &
      fmtf(order_gh)//' (min '//fmtf(MIN_ORDER)//'), truncation growth '//fmtf(trunc(2) / trunc(1))//' (max '// &
      fmtf(MAX_TRUNCATION_GROWTH)//')'//merge('  PASS', '  FAIL', ok)
   call verdict(label, ok, 'iterations '//itoa(its(1))//', '//itoa(its(2))//', symmetry '//fmt(maxval(sym))// &
                ' (<= '//fmt(TOL_SYMMETRY)//'), order '//fmtf(order)//', ghosts '//fmtf(order_gh)//' (>= '//   &
                fmtf(MIN_ORDER)//'), trunc '//fmtf(trunc(1))//' growth '//fmtf(trunc(2) / trunc(1))//' (<= '// &
                fmtf(MAX_TRUNCATION_GROWTH)//'), unfilled '//itoa(sum(unfilled)))
   endsubroutine check_bc_amr

   subroutine check_sub(label, nk, level, amr, bc, isolated)
   !< E5c: one configuration solved with the levels inside the blocks forced (`root_cells=1`) and without them
   !< (`root_cells=huge`).
   character(*), intent(in)           :: label    !< Label.
   integer(I4P), intent(in)           :: nk       !< Cells per block along z.
   integer(I4P), intent(in)           :: level    !< Base level.
   logical,      intent(in)           :: amr      !< Refine the central box (the star's, isolated).
   integer(I4P), intent(in), optional :: bc(2,3)  !< Kinds per face (default periodic).
   logical,      intent(in), optional :: isolated !< Isolated boundaries (the star's cube).
   type(grid_t)                            :: gr    !< Grid.
   type(mg_box)           :: mg(2) !< Solvers: the levels forced, none (see mg_box).
   real(R_P), allocatable :: buf(:), phi(:,:,:,:,:) !< Source, the two potentials.
   real(R_P)              :: d(2)  !< max |Phi_sub - Phi_exact|, max |Phi_exact|.
   integer(I4P)           :: its(2), nsub_expected, s, b, i, j, k !< Iterations, levels, counters.
   logical                :: iso, ok  !< Isolated, pass flag.

   call new_mg(mg(1)%s) ; call new_mg(mg(2)%s)
   iso = .false. ; if (present(isolated)) iso = isolated
   if (iso) then
      per3 = [.false., .false., .false.] ! named: no argument temporary
      call build(gr, 8_I4P, level, merge(REF_STAR, REF_NONE, amr), .true., per3, stat, msg)
      if (failed('build')) return
      buf = source(gr, SRC_STAR, PROFILE_POLYTROPE)
   elseif (present(bc)) then
      per3 = per_bc(bc, nk) ! named: no argument temporary
      call build(gr, nk, level, merge(REF_BOX, REF_NONE, amr), .false., per3, stat, msg)
      if (failed('build')) return
      buf = source(gr, SRC_RHO)
   else
      per3 = per_all(nk) ! named: no argument temporary
      call build(gr, nk, level, merge(REF_BOX, REF_NONE, amr), .false., per3, stat, msg)
      if (failed('build')) return
      buf = source(gr, SRC_RHO)
   endif
   allocate(phi(1-NGC:NB_SIDE+NGC,1-NGC:NB_SIDE+NGC,1-NGC:nk+NGC,1:max(gr%nb,1),2))
   phi = 0._R_P
   do s=1, 2
      if (iso) then
         call mg(s)%s%setup(gr%mesh, c, stat, msg, tolerance=TOL_ISOLATED, max_iterations=MAX_ITERATIONS, &
                          isolated=.true., root_cells=merge(1_I4P, huge(1_I4P), s == 1))
      elseif (present(bc)) then
         call mg(s)%s%setup(gr%mesh, c, stat, msg, max_iterations=MAX_ITERATIONS, bc=bc, &
                          root_cells=merge(1_I4P, huge(1_I4P), s == 1))
      else
         call mg(s)%s%setup(gr%mesh, c, stat, msg, max_iterations=MAX_ITERATIONS, &
                          root_cells=merge(1_I4P, huge(1_I4P), s == 1))
      endif
      if (failed('mg setup')) return
      call mg(s)%s%solve_packed(gr%mesh, buf, FACTOR, phi(:,:,:,:,s), stat, msg) ; if (failed('mg solve')) return
      its(s) = mg(s)%s%iterations + mg(s)%s%iterations0
   enddo
   d = 0._R_P
   do b=1, gr%nb
      do k=1, nk
         do j=1, NB_SIDE
            do i=1, NB_SIDE
               d(1) = max(d(1), abs(phi(i,j,k,b,1) - phi(i,j,k,b,2)))
               d(2) = max(d(2), abs(phi(i,j,k,b,2)))
            enddo
         enddo
      enddo
   enddo
   call c%max_real8(d, stat)
   nsub_expected = nint(log(real(NB_SIDE, R_P)) / log(2._R_P))
   ok = mg(1)%s%lmin - mg(1)%s%lbot == nsub_expected .and. mg(2)%s%lbot == mg(2)%s%lmin
   ok = ok .and. mg(1)%s%residual <= mg(1)%s%tolerance_used .and. mg(2)%s%residual <= mg(2)%s%tolerance_used
   ok = ok .and. d(1) <= TOL_SUB * d(2)
   ok = ok .and. its(1) <= max(MAX_SUB_UNIFORM, nint(MAX_SUB_RATIO * its(2)))
   if (c%rank == 0) print '(A)', '   '//label//' levels inside the blocks '//itoa(mg(1)%s%lmin - mg(1)%s%lbot)// &
      ', iterations '//itoa(its(1))//' (exact coarsest '//itoa(its(2))//'), |Phi_sub - Phi_exact| / max |Phi| '// &
      fmt(d(1) / d(2))//'  '//merge('PASS', 'FAIL', ok)
   call verdict(label, ok, 'levels '//itoa(mg(1)%s%lmin - mg(1)%s%lbot)//' (= '//itoa(nsub_expected)//'), iterations '// &
                itoa(its(1))//' ('//itoa(its(2))//', <= '//itoa(max(MAX_SUB_UNIFORM, nint(MAX_SUB_RATIO * its(2))))// &
                '), |sub-exact|/max '//fmt(d(1) / d(2))//' (<= '//fmt(TOL_SUB)//')')
   call mg(1)%s%destroy ; call mg(2)%s%destroy ; call gr%mesh%destroy
   endsubroutine check_sub

   subroutine check_flail_forward(label, bc, nk, level, amr, dg_order)
   !< E1c: the facade's `setup(..., bc)` and `solve_packed` against the solver it chooses called directly, bit for bit on
   !< every cell and ghost of phi; on AMR (the corner refined) with the manufactured boundary values as `ghost`, the
   !< facade's from its `boundary_faces`, the direct solver's from its own list.
   character(*), intent(in)           :: label    !< Label.
   integer(I4P), intent(in)           :: bc(2,3)  !< Kinds per face [lo/hi, axis].
   integer(I4P), intent(in)           :: nk       !< Cells per block along z.
   integer(I4P), intent(in)           :: level    !< Base level.
   logical,      intent(in)           :: amr      !< Refine the corner (the hierarchical path), else the spectral path.
   integer(I4P), intent(in), optional :: dg_order !< The wide D(G) of this order (spectral path), else compact.
   type(grid_t)                            :: gr  !< Grid.
   type(elisha_solver_object)              :: fl  !< The facade.
   class(elisha_hierarchical_solver_object), allocatable :: mg  !< Multigrid solver, direct.
   class(elisha_spectral_solver_object), allocatable :: fft !< Spectral solver, direct.
   real(R_P),    allocatable :: buf(:), phi_f(:,:,:,:), phi_d(:,:,:,:), ghost_f(:), ghost_d(:), x(:,:) !< Fields.
   integer(I4P), allocatable :: kinds(:), dir(:) !< The facade's boundary faces.
   integer(I4P)              :: ndiff(1), i      !< Values that differ, counter.
   logical                   :: ok               !< Pass flag.
   character(:), allocatable :: note             !< The direct hierarchical solve's convergence (reported).

   call new_mg(mg)
   call new_fft(fft)
   note = ''
   per3 = per_bc(bc, nk) ! named: no argument temporary
   call build(gr, nk, level, merge(REF_CORNER, REF_NONE, amr), .false., per3, stat, msg)
   if (failed('build')) return
   buf = source(gr, SRC_RHO)
   call alloc_phi(gr, phi_f) ; call alloc_phi(gr, phi_d)
   call fl_setup(fl, gr%mesh, stat, msg, bc=bc, dg_order=dg_order) ; if (failed('facade setup')) return
   ok = fl%is_hierarchical() .eqv. amr
   if (amr) then
      call mg%setup(gr%mesh, c, stat, msg, bc=bc) ; if (failed('mg setup')) return
      call fl%boundary_faces(kinds, dir, x, stat, msg) ; if (failed('boundary_faces')) return
      allocate(ghost_f(max(size(kinds),1)), ghost_d(max(mg%nbnd,1)))
      ghost_f = 0._R_P ; ghost_d = 0._R_P
      do i=1, size(kinds)
         ghost_f(i) = face_value(kinds(i), dir(i), x(:,i), gr%active)
      enddo
      do i=1, mg%nbnd
         ghost_d(i) = face_value(mg%bnd_kind(i), mg%bnd_dir(i), mg%bnd_x(:,i), gr%active)
      enddo
      call fl%solve_packed(gr%mesh, buf, FACTOR, phi_f, stat, msg, ghost=ghost_f) ; if (failed('facade solve')) return
      call mg%solve_packed(gr%mesh, buf, FACTOR, phi_d, stat, msg, ghost=ghost_d) ; if (failed('mg solve')) return
      ! reported, not asserted (ADAM's leg asserts the bitwise equality only): the 2-D N/D P/P case does not converge
      note = ', FCG '//itoa(mg%iterations)//' iterations, residual '//fmt(mg%residual)//' (tolerance '// &
             fmt(mg%tolerance_used)//')'
      call mg%destroy
   else
      call fft%setup(gr%mesh, c, stat, msg, bc=bc, dg_order=dg_order) ; if (failed('fft setup')) return
      call fl%solve_packed(gr%mesh, buf, FACTOR, phi_f, stat, msg) ; if (failed('facade solve')) return
      call fft%solve_packed(gr%mesh, buf, FACTOR, phi_d, stat, msg) ; if (failed('fft solve')) return
      call fft%destroy
   endif
   ndiff = bits_differ(phi_f, phi_d)
   call isum(ndiff)
   ok = ok .and. ndiff(1) == 0
   if (c%rank == 0) print '(A)', '   '//label//' path '//trim(merge('multigrid', 'FFT      ', fl%is_hierarchical()))// &
      ', values differing from the direct solve (every cell and ghost, bit for bit) '//itoa(ndiff(1))//note//       &
      merge('  PASS', '  FAIL', ok)
   call verdict(label, ok, 'path '//trim(merge('hierarchical', 'spectral    ', fl%is_hierarchical()))// &
                ', values differing '//itoa(ndiff(1))//' (= 0)'//note)
   call fl%destroy ; call gr%mesh%destroy
   endsubroutine check_flail_forward

   subroutine check_flail_vector(label, amr)
   !< E1c: a vector with per-component kinds, a PEC face normal to z (x, y periodic): Dirichlet for the tangential
   !< components, Neumann for the normal one. One facade per component; the three set up together and solved
   !< interleaved (z, x, y) equal three solves each by a facade of its own, set up and destroyed in turn, bit for bit.
   character(*), intent(in) :: label !< Label.
   logical,      intent(in) :: amr   !< Refine the corner (the hierarchical path), else uniform (the spectral path).
   integer(I4P), parameter  :: ORDER(3)=[3_I4P, 1_I4P, 2_I4P] !< The interleaved order.
   type(grid_t)               :: gr    !< Grid.
   type(elisha_solver_object) :: fl(3) !< One facade per component.
   type(elisha_solver_object) :: one   !< A facade of its own.
   real(R_P), allocatable     :: buf(:), phi_v(:,:,:,:,:), phi_1(:,:,:,:) !< Source, fields.
   integer(I4P)               :: bcv(2,3,3) !< Kinds per component.
   integer(I4P)               :: ndiff(1), cc !< Values that differ, counter.
   logical                    :: ok    !< Pass flag.

   bcv(:,:,1) = BC_PEC_XY ; bcv(:,:,2) = BC_PEC_XY
   bcv(:,:,3) = reshape([P_, P_, P_, P_, N_, N_], [2, 3])
   per3 = per_bc(BC_PEC_XY, 8_I4P) ! named: no argument temporary
   call build(gr, 8_I4P, merge(2_I4P, 1_I4P, amr), merge(REF_CORNER, REF_NONE, amr), .false., per3, stat, msg)
   if (failed('build')) return
   buf = source(gr, SRC_RHO)
   allocate(phi_v(1-NGC:NB_SIDE+NGC,1-NGC:NB_SIDE+NGC,1-NGC:NB_SIDE+NGC,1:max(gr%nb,1),3))
   call alloc_phi(gr, phi_1)
   phi_v = SENTINEL
   ok = .true.
   do cc=1, 3
      call fl_setup(fl(cc), gr%mesh, stat, msg, bc=bcv(:,:,cc)) ; if (failed('facade setup')) return
      ok = ok .and. (fl(cc)%is_hierarchical() .eqv. amr)
   enddo
   do cc=1, 3 ! the factor differs per component: three distinct problems from one source
      call fl(ORDER(cc))%solve_packed(gr%mesh, buf, FACTOR * ORDER(cc), phi_v(:,:,:,:,ORDER(cc)), stat, msg)
      if (failed('facade solve')) return
   enddo
   ndiff = 0
   do cc=1, 3
      phi_1 = SENTINEL
      call fl_setup(one, gr%mesh, stat, msg, bc=bcv(:,:,cc)) ; if (failed('facade setup (alone)')) return
      call one%solve_packed(gr%mesh, buf, FACTOR * cc, phi_1, stat, msg) ; if (failed('facade solve (alone)')) return
      call one%destroy
      ndiff = ndiff + bits_differ(phi_v(:,:,:,:,cc), phi_1)
   enddo
   call isum(ndiff)
   ok = ok .and. ndiff(1) == 0
   if (c%rank == 0) print '(A)', '   '//label//' values differing, interleaved against alone (every cell and ghost) '// &
      itoa(ndiff(1))//merge('  PASS', '  FAIL', ok)
   call verdict(label, ok, 'values differing '//itoa(ndiff(1))//' (= 0)')
   do cc=1, 3
      call fl(cc)%destroy
   enddo
   call gr%mesh%destroy
   endsubroutine check_flail_vector

   subroutine check_refusal(name)
   !< A configuration ELISHA must refuse: the expected status, the same on every rank.
   character(*), intent(in) :: name !< The refusal (ADAM's flag without --refuse-).
   type(grid_t)                            :: gr  !< Grid.
   class(elisha_hierarchical_solver_object), allocatable :: mg  !< Multigrid solver.
   type(elisha_solver_object)              :: fl  !< The facade.
   real(R_P), allocatable :: buf(:), phi(:,:,:,:) !< Source, potential.
   integer(I4P)           :: expected, st(1), lo(1), hi(1) !< Expected status, status and its range over the ranks.
   logical                :: ok   !< Pass flag.
   character(:), allocatable :: label, m_ !< Label, message.

   call new_mg(mg)
   st = ELISHA_OK
   select case(name)
   case('mixed') ! a grid periodic along x only, no bc
      label = 'mixed periodic and bounded axes without bc' ; expected = ELISHA_ERR_BC_MISSING
      per3 = [.true., .false., .false.] ! named: no argument temporary
      call build(gr, 8_I4P, 1_I4P, REF_NONE, .false., per3, st(1), m_)
      if (st(1) == ELISHA_OK) call mg%setup(gr%mesh, c, st(1), m_)
   case('bc-mismatch') ! the same grid, bc periodic along y as well
      label = 'bc periodic along an axis the grid bounds' ; expected = ELISHA_ERR_BC_PERIODICITY
      per3 = [.true., .false., .false.] ! named: no argument temporary
      call build(gr, 8_I4P, 1_I4P, REF_NONE, .false., per3, st(1), m_)
      if (st(1) == ELISHA_OK) call mg%setup(gr%mesh, c, st(1), m_, bc=reshape([P_, P_, P_, P_, D_, D_], [2, 3]))
   case('bc-mixed-ghost') ! the same grid, a ghost-Dirichlet face and a Dirichlet one along y
      label = 'a ghost-Dirichlet face against a face kind' ; expected = ELISHA_ERR_BC_GHOST_PAIR
      per3 = [.true., .false., .false.] ! named: no argument temporary
      call build(gr, 8_I4P, 1_I4P, REF_NONE, .false., per3, st(1), m_)
      if (st(1) == ELISHA_OK) call mg%setup(gr%mesh, c, st(1), m_, bc=reshape([P_, P_, G_, D_, D_, D_], [2, 3]))
   case('boundary-refined') ! the isolated cube at level 1, the octant x, y, z > 0 refined
      label = 'isolated, a refined leaf on the boundary' ; expected = ELISHA_ERR_ISOLATED_BOUNDARY_LEAF
      per3 = [.false., .false., .false.] ! named: no argument temporary
      call build(gr, 8_I4P, 1_I4P, REF_OCTANT, .true., per3, st(1), m_)
      if (st(1) == ELISHA_OK) call mg%setup(gr%mesh, c, st(1), m_, isolated=.true.)
   case('flail-ghost-fft') ! boundary values on the spectral path (a uniform grid bounded along z)
      label = 'facade, boundary values on the spectral path' ; expected = ELISHA_ERR_GHOST_VALUES
      per3 = per_bc(BC_PEC_XY, 8_I4P) ! named: no argument temporary
      call build(gr, 8_I4P, 1_I4P, REF_NONE, .false., per3, st(1), m_)
      if (st(1) == ELISHA_OK) call fl_setup(fl, gr%mesh, st(1), m_, bc=BC_PEC_XY)
      if (st(1) == ELISHA_OK) then
         buf = source(gr, SRC_RHO)
         call alloc_phi(gr, phi)
         call fl%solve_packed(gr%mesh, buf, FACTOR, phi, st(1), m_, ghost=[0._R_P])
      endif
   case('flail-isolated-bc') ! isolated with bc
      label = 'facade, isolated with bc' ; expected = ELISHA_ERR_ISOLATED_BC
      per3 = [.false., .false., .false.] ! named: no argument temporary
      call build(gr, 8_I4P, 1_I4P, REF_NONE, .false., per3, st(1), m_)
      if (st(1) == ELISHA_OK) call fl_setup(fl, gr%mesh, st(1), m_, isolated=.true., bc=BC_B)
   case('flail-wide-amr') ! the wide operator on AMR (the corner refined)
      label = 'facade, the wide D(G) on AMR' ; expected = ELISHA_ERR_DG_AMR
      per3 = per_bc(BC_B, 8_I4P) ! named: no argument temporary
      call build(gr, 8_I4P, 2_I4P, REF_CORNER, .false., per3, st(1), m_)
      if (st(1) == ELISHA_OK) call fl_setup(fl, gr%mesh, st(1), m_, bc=BC_B, dg_order=4_I4P)
   endselect
   lo = st ; hi = st
   call c%min_i4(lo, stat) ; call c%max_i4(hi, stat)
   ok = lo(1) == expected .and. hi(1) == expected
   if (c%rank == 0) then
      print '(A)', '   refusal, '//label//': '//elisha_status_name(st(1))//' (expected '//elisha_status_name(expected)// &
                   ')  '//merge('PASS', 'FAIL', ok)
      if (allocated(m_)) print '(A)', '      '//m_
   endif
   call verdict('refusal, '//label//':', ok, elisha_status_name(st(1)))
   call mg%destroy ; call fl%destroy ; call gr%mesh%destroy
   endsubroutine check_refusal

   ! ---------------------------------------------------------------------------------------------------------------------
   ! the measures
   ! ---------------------------------------------------------------------------------------------------------------------
   function symmetry(gr, mg) result(sym)
   !< The operator's symmetry defect for ADAM's two pseudo-random vectors (they depend on the rank and the local order).
   type(grid_t),                            intent(in)    :: gr  !< Grid.
   class(elisha_hierarchical_solver_object), intent(inout) :: mg  !< Solver.
   real(R_P)                                              :: sym !< Defect.
   real(R_P), allocatable :: x(:), y(:) !< Vectors.
   integer(I4P)           :: i, nl      !< Counters.

   nl = gr%nb * ncb(gr)
   allocate(x(max(nl,1)), y(max(nl,1)))
   x = 0._R_P ; y = 0._R_P
   do i=1, nl
      x(i) = sin(1.3_R_P * i + 0.7_R_P * c%rank) ; y(i) = cos(2.1_R_P * i - 0.4_R_P * c%rank)
   enddo
   sym = mg%symmetry_defect(x, y, stat)
   if (stat /= ELISHA_OK) sym = huge(1._R_P)
   endfunction symmetry

   subroutine truncation(gr, mg, tmax, ghost, source_mean)
   !< The operator's truncation error: A applied to the exact potential against V 4 pi G (rho - rho0), the max over the
   !< leaves per unit volume. Dirichlet: A Phi_exact + B g with the ghost values g, against V 4 pi G (rho - mean), the
   !< mean `source_mean` (the source is rho - rho0 already).
   type(grid_t),                            intent(in)    :: gr   !< Grid.
   class(elisha_hierarchical_solver_object), intent(inout) :: mg   !< Solver.
   real(R_P),                               intent(out)   :: tmax !< Max truncation.
   real(R_P), optional,                     intent(in)    :: ghost(1:)   !< Dirichlet: the ghost values.
   real(R_P), optional,                     intent(in)    :: source_mean !< The mean subtracted from rho (default rho0).
   real(R_P), allocatable :: pe(:), ye(:), yb(:) !< Exact potential, the operator on it, B g.
   real(R_P)              :: rmean, xyz(3), v, t(1) !< Measures.
   integer(I4P)           :: b, i, j, k, cc, n   !< Counters.

   n = gr%nb * ncb(gr)
   allocate(pe(max(n,1)), ye(max(n,1)))
   pe = 0._R_P ; ye = 0._R_P
   cc = 0
   do b=1, gr%nb
      do k=1, gr%nk
         do j=1, NB_SIDE
            do i=1, NB_SIDE
               cc = cc + 1
               pe(cc) = phi_exact(cell_centre(gr, b, i, j, k), gr%active)
            enddo
         enddo
      enddo
   enddo
   call mg%apply_a(pe, ye, stat)
   if (present(ghost)) then
      allocate(yb(max(n,1)))
      yb = 0._R_P
      call mg%apply_b(ghost, yb)
      if (n > 0) ye(1:n) = ye(1:n) + yb(1:n)
   endif
   rmean = RHO0
   if (present(source_mean)) rmean = RHO0 - source_mean
   t = 0._R_P
   cc = 0
   do b=1, gr%nb
      v = cell_volume(gr, b)
      do k=1, gr%nk
         do j=1, NB_SIDE
            do i=1, NB_SIDE
               cc = cc + 1
               xyz = cell_centre(gr, b, i, j, k)
               t(1) = max(t(1), abs(ye(cc) / v - FACTOR * (rho(xyz, gr%active) - rmean)))
            enddo
         enddo
      enddo
   enddo
   call c%max_real8(t, stat)
   tmax = t(1)
   endsubroutine truncation

   subroutine measure(gr, phi, e_phi, e_grav, unfilled, periodic)
   !< Max error of Phi against the exact one (leaves); max error of the face gravity on the faces whose neighbour is
   !< coarser (the fine side of 2:1 faces, from the filled ghost); unfilled face ghosts. The neighbour's level comes
   !< from the validated mesh's leaf table.
   type(grid_t), intent(in)  :: gr       !< Grid.
   real(R_P),    intent(in)  :: phi(1-NGC:,1-NGC:,1-NGC:,1:) !< Phi.
   real(R_P),    intent(out) :: e_phi    !< Max |Phi - Phi_exact|.
   real(R_P),    intent(out) :: e_grav   !< Max face-gravity error on 2:1 faces.
   integer(I4P), intent(out) :: unfilled !< Unfilled face ghosts.
   logical,      intent(in)  :: periodic !< Periodic boundaries; else a domain face is no 2:1 face.
   real(R_P)                 :: xyz(3), xf(3), p(3), g_num, g_ex, hd(3), e(2) !< Measures.
   integer(I4P)              :: b, i, j, k, d, s, lv, ijk(3), nb(3), u(1) !< Counters.

   e = 0._R_P ; u = 0
   do b=1, gr%nb
      lv = gr%mesh%coords(4,b)
      hd = cell_dx(gr, b)
      do k=1, gr%nk
         do j=1, NB_SIDE
            do i=1, NB_SIDE
               xyz = cell_centre(gr, b, i, j, k)
               e(1) = max(e(1), abs(phi(i,j,k,b) - phi_exact(xyz, gr%active)))
               do d=1, 3
                  if (.not.gr%active(d)) cycle
                  do s=-1, 1, 2
                     ijk = [i, j, k] ; nb = ijk ; nb(d) = nb(d) + s
                     if (nb(d) >= 1 .and. nb(d) <= nblk(gr, d)) cycle
                     if (phi(nb(1),nb(2),nb(3),b) == SENTINEL) u = u + 1
                     p = xyz ; p(d) = p(d) + s * hd(d)
                     if (.not.periodic .and. (p(d) < gr%emin(d) .or. p(d) > gr%emin(d) + gr%lb(d))) cycle
                     p(d) = gr%emin(d) + modulo(p(d) - gr%emin(d), gr%lb(d))
                     if (level_at(gr, p) >= lv) cycle
                     g_num = s * (phi(nb(1),nb(2),nb(3),b) - phi(i,j,k,b)) / hd(d)
                     xf = xyz ; xf(d) = xf(d) + s * hd(d) / 2._R_P
                     g_ex = grad_exact(xf, gr%active, d)
                     e(2) = max(e(2), abs(g_num - g_ex))
                  enddo
               enddo
            enddo
         enddo
      enddo
   enddo
   call c%max_real8(e, stat)
   call isum(u)
   e_phi = e(1) ; e_grav = e(2) ; unfilled = u(1)
   endsubroutine measure

   subroutine measure_star(gr, phi, prof, e_phi, e_ghost, unfilled)
   !< Max error of Phi against the star's exact potential on the leaves; on the first ghost layer beyond a domain face
   !< (the James ghost values); unfilled face ghosts.
   type(grid_t), intent(in)  :: gr       !< Grid.
   real(R_P),    intent(in)  :: phi(1-NGC:,1-NGC:,1-NGC:,1:) !< Phi.
   integer(I4P), intent(in)  :: prof     !< Source.
   real(R_P),    intent(out) :: e_phi    !< Max |Phi - Phi_exact| on the leaves.
   real(R_P),    intent(out) :: e_ghost  !< Max |Phi - Phi_exact| on the boundary ghosts.
   integer(I4P), intent(out) :: unfilled !< Unfilled face ghosts.
   real(R_P)                 :: xyz(3), p(3), hd(3), e(2) !< Points, spacing, errors.
   integer(I4P)              :: b, i, j, k, d, s, ijk(3), nb(3), u(1) !< Counters.

   e = 0._R_P ; u = 0
   do b=1, gr%nb
      hd = cell_dx(gr, b)
      do k=1, gr%nk
         do j=1, NB_SIDE
            do i=1, NB_SIDE
               xyz = cell_centre(gr, b, i, j, k)
               e(1) = max(e(1), abs(phi(i,j,k,b) - phi_sphere(xyz, prof)))
               do d=1, 3
                  do s=-1, 1, 2
                     ijk = [i, j, k] ; nb = ijk ; nb(d) = nb(d) + s
                     if (nb(d) >= 1 .and. nb(d) <= nblk(gr, d)) cycle
                     if (phi(nb(1),nb(2),nb(3),b) == SENTINEL) u = u + 1
                     p = xyz ; p(d) = p(d) + s * hd(d)
                     if (p(d) > gr%emin(d) .and. p(d) < gr%emin(d) + gr%lb(d)) cycle
                     e(2) = max(e(2), abs(phi(nb(1),nb(2),nb(3),b) - phi_sphere(p, prof)))
                  enddo
               enddo
            enddo
         enddo
      enddo
   enddo
   call c%max_real8(e, stat)
   call isum(u)
   e_phi = e(1) ; e_ghost = e(2) ; unfilled = u(1)
   endsubroutine measure_star

   subroutine interior_diff(gr, p1, p2, d)
   !< max |p1 - p2| and max |p2| over the interior cells, over the ranks. A subroutine, not an array-valued function:
   !< nvfortran 26.9 with -Mbounds (tests-nvf-debug) reports garbage bounds for the function's d(2) result and stops.
   type(grid_t), intent(in)  :: gr   !< Grid.
   real(R_P),    intent(in)  :: p1(1-NGC:,1-NGC:,1-NGC:,1:) !< First potential.
   real(R_P),    intent(in)  :: p2(1-NGC:,1-NGC:,1-NGC:,1:) !< Second potential (the reference).
   real(R_P),    intent(out) :: d(2) !< The two maxima.
   integer(I4P)             :: b, i, j, k !< Counters.

   d = 0._R_P
   do b=1, gr%nb ! explicit loops: no array temporary (ifx puts them on the stack)
      do k=1, gr%nk
         do j=1, NB_SIDE
            do i=1, NB_SIDE
               d(1) = max(d(1), abs(p1(i,j,k,b) - p2(i,j,k,b)))
               d(2) = max(d(2), abs(p2(i,j,k,b)))
            enddo
         enddo
      enddo
   enddo
   call c%max_real8(d, stat)
   endsubroutine interior_diff

   function bits_differ(p1, p2) result(n)
   !< The values of two arrays of one shape whose bit patterns differ (this rank).
   real(R_P), intent(in) :: p1(:,:,:,:) !< First.
   real(R_P), intent(in) :: p2(:,:,:,:) !< Second.
   integer(I4P)          :: n(1)        !< Count.
   integer(I4P)          :: i, j, k, b  !< Counters.

   n = 0 ! explicit loops: no array temporary (ifx puts them on the stack)
   do b=1, size(p1, dim=4)
      do k=1, size(p1, dim=3)
         do j=1, size(p1, dim=2)
            do i=1, size(p1, dim=1)
               if (transfer(p1(i,j,k,b), 0_I8P) /= transfer(p2(i,j,k,b), 0_I8P)) n(1) = n(1) + 1
            enddo
         enddo
      enddo
   enddo
   endfunction bits_differ

   subroutine isum(x)
   !< x <- its elementwise sum over the ranks.
   integer(I4P), intent(inout) :: x(:) !< Values.
   integer(I8P)                :: w(size(x)) !< Buffer.

   w = int(x, I8P)
   call c%sum_i8(w, stat)
   x = int(w, I4P)
   endsubroutine isum

   subroutine allmax(v, a1, a2, a3)
   !< Three scalars to their maxima over the ranks.
   real(R_P), intent(in)  :: v(3)       !< Values.
   real(R_P), intent(out) :: a1, a2, a3 !< Maxima.
   real(R_P)              :: w(3)       !< Buffer.

   w = v
   call c%max_real8(w, stat)
   a1 = w(1) ; a2 = w(2) ; a3 = w(3)
   endsubroutine allmax

   subroutine dump_leaves(gr, phi, name)
   !< With --dump: rank 0 writes the leaves' interior potential, Morton order, in ADAM's dump format.
   type(grid_t), intent(in) :: gr   !< Grid.
   real(R_P),    intent(in) :: phi(1-NGC:,1-NGC:,1-NGC:,1:) !< Potential.
   character(*), intent(in) :: name !< Case name.
   real(R_P),    allocatable :: sbuf(:), rbuf(:), g(:,:) !< Send (leaf id, values per block), receive, global.
   integer(I4P), allocatable :: counts(:), displs(:)      !< Per rank.
   integer(I4P)              :: n, b, i, j, k, p, r, u, ios !< Counters.

   if (len_trim(dump_dir) == 0) return
   n = ncb(gr)
   allocate(sbuf(max(gr%nb*(n+1),1)))
   p = 0
   do b=1, gr%nb
      p = p + 1 ; sbuf(p) = real(gr%gid(b), R_P)
      do k=1, gr%nk
         do j=1, NB_SIDE
            do i=1, NB_SIDE
               p = p + 1 ; sbuf(p) = phi(i,j,k,b)
            enddo
         enddo
      enddo
   enddo
   allocate(counts(c%nranks), displs(c%nranks))
   counts = 0 ; displs = 0
   call c%gather_i4(gr%nb*(n+1), counts, stat)
   if (c%rank == 0) then
      do r=2, c%nranks
         displs(r) = displs(r-1) + counts(r-1)
      enddo
      allocate(rbuf(max(sum(counts),1)))
   else
      allocate(rbuf(1))
   endif
   call c%gatherv_r(sbuf, gr%nb*(n+1), rbuf, counts, displs, stat)
   if (c%rank /= 0) return
   allocate(g(n,gr%nleaf))
   g = SENTINEL
   p = 0
   do b=1, gr%nleaf
      i = nint(rbuf(p+1))
      g(:,i) = rbuf(p+2:p+1+n)
      p = p + n + 1
   enddo
   open(newunit=u, file=trim(dump_dir)//'/phi-mg-'//name//'-l'//itoa(gr%level)//'.bin', access='stream', &
        form='unformatted', action='write', status='replace', iostat=ios)
   if (ios /= 0) then
      print '(A)', '   dump: cannot open the file of '//name
      test_passed = .false.
      return
   endif
   write(u) [n, gr%nleaf, 1_I4P]
   write(u) g
   close(u)
   endsubroutine dump_leaves

   logical function failed(what)
   !< A step returned an error: report it, fail the test.
   character(*), intent(in) :: what !< The step.

   failed = stat /= ELISHA_OK
   if (.not.failed) return
   test_passed = .false.
   if (c%rank == 0) then
      if (allocated(msg)) then
         print '(A)', '   '//what//' failed: '//msg
      else
         print '(A)', '   '//what//' failed: '//elisha_status_name(stat)
      endif
   endif
   call add_row(what//' failed: '//elisha_status_name(stat)//' | FAIL')
   endfunction failed

   ! ---------------------------------------------------------------------------------------------------------------------
   ! the grids
   ! ---------------------------------------------------------------------------------------------------------------------
   subroutine build(gr, nk, level, refine, cube, periodic, st, m_)
   !< ADAM's grid: one root block, uniform at `level`, the blocks marked by `refine` refined once; the leaves in Morton
   !< order, dealt in contiguous chunks to the ranks; the mesh of this rank's share.
   type(grid_t),              intent(inout) :: gr          !< Grid.
   integer(I4P),              intent(in)    :: nk          !< Cells per block along z (1: null z).
   integer(I4P),              intent(in)    :: level       !< Base level.
   integer(I4P),              intent(in)    :: refine      !< REF_*.
   logical,                   intent(in)    :: cube        !< The cube [-1,1]^3.
   logical,                   intent(in)    :: periodic(3) !< Periodic axes of the mesh.
   integer(I4P),              intent(out)   :: st          !< Status.
   character(:), allocatable, intent(out)   :: m_          !< Message.
   integer(I4P), allocatable :: tmp(:,:)  !< Leaves before the sort.
   integer(I8P), allocatable :: key(:)    !< Morton keys.
   integer(I4P)              :: nbl(3), n, i, j, k, o, lmax, lo, hi, g, t4(4) !< Counters.
   integer(I8P)              :: tk        !< A key.
   integer(I4P)              :: sc(3), nijk(3), nblk1(3) !< Scaled coordinates, cells per block, blocks at level 0.
   real(R_P)                 :: dx0(3), emin3(3) !< Spacing at level 0, lower corner (named actual arguments).
   logical                   :: per(3), null3(3) !< Periodic and null axes (named actual arguments).

   call gr%mesh%destroy
   gr%nk = nk ; gr%level = level ; gr%cube = cube
   gr%null = [.false., .false., nk == 1] ; gr%active = .not.gr%null
   call domain(cube, gr%lb, gr%emin)
   nbl = [2_I4P**level, 2_I4P**level, merge(1_I4P, 2_I4P**level, nk == 1)]
   allocate(tmp(4,product(nbl)*8))
   n = 0
   do k=0, nbl(3)-1
      do j=0, nbl(2)-1
         do i=0, nbl(1)-1
            if (marked(gr, i, j, k, level, refine)) then
               do o=0, 7
                  if (gr%null(3) .and. ibits(o, 2, 1) == 1) cycle
                  n = n + 1
                  tmp(:,n) = [2*i + ibits(o, 0, 1), 2*j + ibits(o, 1, 1), merge(0_I4P, 2*k + ibits(o, 2, 1), gr%null(3)), &
                              level + 1]
               enddo
            else
               n = n + 1
               tmp(:,n) = [i, j, k, level]
            endif
         enddo
      enddo
   enddo
   lmax = maxval(tmp(4,1:n))
   allocate(key(n))
   do i=1, n
      sc = tmp(1:3,i) * 2_I4P**(lmax - tmp(4,i)) ! named: no argument temporary
      key(i) = morton(sc)
   enddo
   do i=2, n ! insertion sort by key (a few hundred leaves)
      tk = key(i) ; t4 = tmp(:,i)
      j = i - 1
      do while (j >= 1)
         if (key(j) <= tk) exit
         key(j+1) = key(j) ; tmp(:,j+1) = tmp(:,j)
         j = j - 1
      enddo
      key(j+1) = tk ; tmp(:,j+1) = t4
   enddo
   gr%nleaf = n
   gr%leaves = tmp(:,1:n)
   lo = int((int(c%rank, I8P) * n) / c%nranks, I4P)
   hi = int((int(c%rank + 1, I8P) * n) / c%nranks, I4P)
   gr%nb = hi - lo
   gr%gid = [(g, g=lo+1, hi)]
   nijk = [NB_SIDE, NB_SIDE, nk] ; nblk1 = 1_I4P ; dx0 = gr%lb / real(nijk, R_P) ; per = periodic ! named arguments
   null3 = gr%null ; emin3 = gr%emin
   call gr%mesh%init(nijk, NGC, nblk1, dx0, emin3, per, null3, gr%leaves(:,lo+1:hi), st, m_)
   endsubroutine build

   pure function morton(ijk) result(key)
   !< The Morton key of block coordinates (x fastest).
   integer(I4P), intent(in) :: ijk(3) !< Coordinates.
   integer(I8P)             :: key    !< Key.
   integer(I4P)             :: bt, d  !< Counters.

   key = 0_I8P
   do bt=0, 19
      do d=1, 3
         if (btest(ijk(d), bt)) key = ibset(key, 3 * bt + d - 1)
      enddo
   enddo
   endfunction morton

   pure function marked(gr, i, j, k, lv, refine) result(mk)
   !< ADAM's marks: a block refined by its centre (the block's corner and size in ADAM's arithmetic).
   type(grid_t), intent(in) :: gr          !< Grid.
   integer(I4P), intent(in) :: i, j, k, lv !< Block (i, j, k, level), scalars (no argument temporary).
   integer(I4P), intent(in) :: refine      !< REF_*.
   integer(I4P)             :: c4(4)       !< Block.
   logical                  :: mk     !< Marked.
   real(R_P)                :: bdx(3), bmin(3), ctr(3) !< Block size, corner, centre.

   c4 = [i, j, k, lv]
   bdx = block_dx(gr, c4(4))
   bmin = gr%emin + c4(1:3) * bdx
   ctr = 0.5_R_P * (bmin + (bmin + bdx))
   select case(refine)
   case(REF_BOX)
      mk = all(abs(ctr - 0.5_R_P * L) < 0.25_R_P * L .or. gr%null)
   case(REF_CORNER)
      mk = all(ctr < 0.5_R_P * L .or. gr%null)
   case(REF_STAR)
      mk = maxval(abs(ctr)) < A_SPHERE
   case(REF_OCTANT)
      mk = all(ctr > 0._R_P)
   case default
      mk = .false.
   endselect
   endfunction marked

   pure subroutine domain(cube, lb, emin)
   !< The domain: the cube [-1,1]^3 or the box [0,1] x [0,2] x [0,3].
   logical,   intent(in)  :: cube    !< The cube.
   real(R_P), intent(out) :: lb(3)   !< Lengths.
   real(R_P), intent(out) :: emin(3) !< Lower corner.

   if (cube) then
      lb = 2._R_P ; emin = -1._R_P
   else
      lb = L ; emin = 0._R_P
   endif
   endsubroutine domain

   pure function block_dx(gr, level) result(bdx)
   !< ADAM's block size at a level: the domain over 2^level along an active axis (the whole length along a null one).
   type(grid_t), intent(in) :: gr     !< Grid.
   integer(I4P), intent(in) :: level  !< Level.
   real(R_P)                :: bdx(3) !< Block size.

   bdx = gr%lb / 2_I4P**level
   if (gr%null(3)) bdx(3) = gr%lb(3)
   endfunction block_dx

   pure function cell_dx(gr, b) result(cdx)
   !< ADAM's cell spacing of this rank's block b (grid_object%cell_dxyz at its level).
   type(grid_t), intent(in) :: gr     !< Grid.
   integer(I4P), intent(in) :: b      !< Block.
   real(R_P)                :: cdx(3) !< Spacing.

   cdx = block_dx(gr, gr%mesh%coords(4,b)) / [NB_SIDE, NB_SIDE, gr%nk]
   endfunction cell_dx

   pure function cell_volume(gr, b) result(v)
   !< ADAM's cell volume of block b: the product of the active spacings.
   type(grid_t), intent(in) :: gr !< Grid.
   integer(I4P), intent(in) :: b  !< Block.
   real(R_P)                :: v  !< Volume.

   v = product(merge(cell_dx(gr, b), 1._R_P, gr%active))
   endfunction cell_volume

   pure function cell_centre(gr, b, i, j, k) result(xyz)
   !< The centre of cell (i, j, k) (ghosts included) of this rank's block b, in ADAM's arithmetic
   !< (grid_object%cell_xyz). Scalar indices: an array constructor per call is an argument temporary, which ifx's
   !< `-check arg_temp_created` reports at every call.
   type(grid_t), intent(in) :: gr      !< Grid.
   integer(I4P), intent(in) :: b       !< Block.
   integer(I4P), intent(in) :: i, j, k !< Cell.
   real(R_P)                :: xyz(3)  !< Centre.
   real(R_P)                :: bdx(3), cdx(3), bmin(3) !< Block size, cell size, block corner.
   integer(I4P)             :: nijk(3), ijk(3) !< Cells per block, the cell.

   ijk = [i, j, k]
   nijk = [NB_SIDE, NB_SIDE, gr%nk]
   bdx = block_dx(gr, gr%mesh%coords(4,b))
   cdx = bdx / nijk
   bmin = gr%emin + gr%mesh%coords(1:3,b) * bdx
   xyz = bmin - bdx / nijk * 0.5_R_P + ijk * cdx
   endfunction cell_centre

   pure function level_at(gr, p) result(lv)
   !< The level of the leaf containing point p (inside the domain), from the validated mesh's leaf table (-1: none).
   type(grid_t), intent(in) :: gr    !< Grid.
   real(R_P),    intent(in) :: p(3)  !< Point.
   integer(I4P)             :: lv    !< Level.
   integer(I4P)             :: l, c4(4) !< Level, block.

   lv = -1
   do l=gr%mesh%lmin, gr%mesh%lmax
      c4(1:3) = merge(0_I4P, int(floor((p - gr%emin) / block_dx(gr, l)), I4P), gr%null)
      c4(4) = l
      if (gr%mesh%leaf_of(c4) > 0) then
         lv = l
         return
      endif
   enddo
   endfunction level_at

   pure integer(I4P) function ncb(gr)
   !< Cells per block.
   type(grid_t), intent(in) :: gr !< Grid.

   ncb = NB_SIDE * NB_SIDE * gr%nk
   endfunction ncb

   pure integer(I4P) function nblk(gr, d)
   !< Cells per block along axis d.
   type(grid_t), intent(in) :: gr !< Grid.
   integer(I4P), intent(in) :: d  !< Axis.

   nblk = NB_SIDE
   if (d == 3) nblk = gr%nk
   endfunction nblk

   pure function per_all(nk) result(p)
   !< Every active axis periodic.
   integer(I4P), intent(in) :: nk   !< Cells per block along z.
   logical                  :: p(3) !< Periodic axes.

   p = [.true., .true., nk > 1]
   endfunction per_all

   pure function per_bc(bc, nk) result(p)
   !< ADAM's set_grid_bc: an axis periodic in `bc` (and active) is periodic in the mesh, the others bounded.
   integer(I4P), intent(in) :: bc(2,3) !< Kinds.
   integer(I4P), intent(in) :: nk      !< Cells per block along z.
   logical                  :: p(3)    !< Periodic axes.
   integer(I4P)             :: d       !< Counter.

   do d=1, 3
      p(d) = all(bc(:,d) == ELISHA_BC_PERIODIC) .and. .not.(d == 3 .and. nk == 1)
   enddo
   endfunction per_bc

   subroutine alloc_phi(gr, phi)
   !< A potential with every ghost layer, the sentinel everywhere.
   type(grid_t),           intent(in)    :: gr       !< Grid.
   real(R_P), allocatable, intent(inout) :: phi(:,:,:,:) !< Potential.

   if (allocated(phi)) deallocate(phi)
   allocate(phi(1-NGC:NB_SIDE+NGC,1-NGC:NB_SIDE+NGC,1-NGC:gr%nk+NGC,1:max(gr%nb,1)))
   phi = SENTINEL
   endsubroutine alloc_phi

   function source(gr, kind, prof) result(buf)
   !< The packed source of this rank's blocks (block-major, i fastest).
   type(grid_t),           intent(in) :: gr   !< Grid.
   integer(I4P),           intent(in) :: kind !< SRC_*.
   integer(I4P), optional, intent(in) :: prof !< The star's profile (SRC_STAR).
   real(R_P), allocatable             :: buf(:) !< Source.
   real(R_P)                          :: xyz(3) !< Cell centre.
   integer(I4P)                       :: b, i, j, k, p !< Counters.

   allocate(buf(max(gr%nb*ncb(gr),1)))
   buf = 0._R_P
   p = 0
   do b=1, gr%nb
      do k=1, gr%nk
         do j=1, NB_SIDE
            do i=1, NB_SIDE
               p = p + 1
               xyz = cell_centre(gr, b, i, j, k)
               select case(kind)
               case(SRC_RHO)
                  buf(p) = rho(xyz, gr%active)
               case(SRC_RHO_DEV)
                  buf(p) = rho(xyz, gr%active) - RHO0
               case(SRC_STAR)
                  buf(p) = rho_sphere(xyz, prof, cell_dx(gr, b))
               case(SRC_LAPM)
                  buf(p) = lap_m(xyz, gr%active) / FACTOR
               endselect
            enddo
         enddo
      enddo
   enddo
   endfunction source

   ! ---------------------------------------------------------------------------------------------------------------------
   ! the exact and manufactured solutions (ADAM's)
   ! ---------------------------------------------------------------------------------------------------------------------
   pure function wave(m, active) result(kv)
   !< Wave vector of mode m; none along an inactive direction.
   integer(I4P), intent(in) :: m         !< Mode.
   logical,      intent(in) :: active(3) !< Active directions.
   real(R_P)                :: kv(3)     !< Wave vector.

   if (m == 1) then
      kv = 2._R_P * PI * [1._R_P, 2._R_P, 1._R_P] / L
   else
      kv = 2._R_P * PI * [2._R_P, 1._R_P, 1._R_P] / L
   endif
   where (.not.active) kv = 0._R_P
   endfunction wave

   pure function rho(xyz, active) result(r)
   !< The source density.
   real(R_P), intent(in) :: xyz(3)       !< Point.
   logical,   intent(in) :: active(3)    !< Active directions.
   real(R_P)             :: r            !< Density.
   real(R_P)             :: k1(3), k2(3) !< Wave vectors.

   k1 = wave(1_I4P, active) ; k2 = wave(2_I4P, active)
   r = RHO0 + A1 * sin(k1(1) * xyz(1)) * cos(k1(2) * xyz(2)) * cos(k1(3) * xyz(3)) + A2 * cos(dot_product(k2, xyz))
   endfunction rho

   pure function phi_exact(xyz, active) result(p)
   !< The exact (continuum) potential, zero mean.
   real(R_P), intent(in) :: xyz(3)       !< Point.
   logical,   intent(in) :: active(3)    !< Active directions.
   real(R_P)             :: p            !< Potential.
   real(R_P)             :: k1(3), k2(3) !< Wave vectors.

   k1 = wave(1_I4P, active) ; k2 = wave(2_I4P, active)
   p = FACTOR * (A1 * sin(k1(1) * xyz(1)) * cos(k1(2) * xyz(2)) * cos(k1(3) * xyz(3)) / (-dot_product(k1, k1)) + &
                 A2 * cos(dot_product(k2, xyz)) / (-dot_product(k2, k2)))
   endfunction phi_exact

   pure function grad_exact(xyz, active, d) result(gr)
   !< d Phi / d x_d of the exact potential.
   real(R_P),    intent(in) :: xyz(3)              !< Point.
   logical,      intent(in) :: active(3)           !< Active directions.
   integer(I4P), intent(in) :: d                   !< Direction.
   real(R_P)                :: gr                  !< Derivative.
   real(R_P)                :: k1(3), k2(3), t1(3) !< Wave vectors, factors.

   k1 = wave(1_I4P, active) ; k2 = wave(2_I4P, active)
   t1 = [sin(k1(1) * xyz(1)), cos(k1(2) * xyz(2)), cos(k1(3) * xyz(3))]
   select case(d)
   case(1)
      gr = k1(1) * cos(k1(1) * xyz(1)) * t1(2) * t1(3)
   case(2)
      gr = -k1(2) * t1(1) * sin(k1(2) * xyz(2)) * t1(3)
   case default
      gr = -k1(3) * t1(1) * t1(2) * sin(k1(3) * xyz(3))
   endselect
   gr = FACTOR * (A1 * gr / (-dot_product(k1, k1)) - A2 * k2(d) * sin(dot_product(k2, xyz)) / (-dot_product(k2, k2)))
   endfunction grad_exact

   pure function phi_m(xyz, active) result(p)
   !< The manufactured potential, prod_d cos(2 pi x_d / L_d + theta_d) over the active directions.
   real(R_P), intent(in) :: xyz(3)    !< Point.
   logical,   intent(in) :: active(3) !< Active directions.
   real(R_P)             :: p         !< Potential.
   integer(I4P)          :: d         !< Counter.

   p = 1._R_P
   do d=1, 3
      if (active(d)) p = p * cos(2._R_P * PI * xyz(d) / L(d) + THETA(d))
   enddo
   endfunction phi_m

   pure function lap_m(xyz, active) result(lp)
   !< The Laplacian of the manufactured potential, -sum_d (2 pi / L_d)^2 phi_m.
   real(R_P), intent(in) :: xyz(3)    !< Point.
   logical,   intent(in) :: active(3) !< Active directions.
   real(R_P)             :: lp        !< Laplacian.

   lp = -sum(merge((2._R_P * PI / L)**2, 0._R_P, active)) * phi_m(xyz, active)
   endfunction lap_m

   pure function grad_m(xyz, active, d) result(gr)
   !< d phi_m / d x_d.
   real(R_P),    intent(in) :: xyz(3)    !< Point.
   logical,      intent(in) :: active(3) !< Active directions.
   integer(I4P), intent(in) :: d         !< Direction.
   real(R_P)                :: gr        !< Derivative.
   integer(I4P)             :: e         !< Counter.

   gr = -2._R_P * PI / L(d) * sin(2._R_P * PI * xyz(d) / L(d) + THETA(d))
   do e=1, 3
      if (active(e) .and. e /= d) gr = gr * cos(2._R_P * PI * xyz(e) / L(e) + THETA(e))
   enddo
   endfunction grad_m

   pure function face_value(kind, dir, x, active) result(g)
   !< The manufactured boundary value of a face of a kind: the outward derivative (Neumann), else the value at x.
   integer(I4P), intent(in) :: kind      !< Kind.
   integer(I4P), intent(in) :: dir       !< Outward axis, signed.
   real(R_P),    intent(in) :: x(3)      !< Where the value lives.
   logical,      intent(in) :: active(3) !< Active directions.
   real(R_P)                :: g         !< Value.

   if (kind == ELISHA_BC_NEUMANN) then
      g = sign(1_I4P, dir) * grad_m(x, active, abs(dir))
   else
      g = phi_m(x, active)
   endif
   endfunction face_value

   pure function rho_sphere(xyz, prof, dxyz) result(r)
   !< The isolated source: the n = 1 polytrope at the point, or the uniform sphere's cell average (SUB^3 samples).
   real(R_P),    intent(in) :: xyz(3)  !< Cell centre.
   integer(I4P), intent(in) :: prof    !< Source.
   real(R_P),    intent(in) :: dxyz(3) !< Cell spacing.
   real(R_P)                :: r       !< Density.
   real(R_P)                :: kr      !< k r.
   real(R_P)                :: p(3)    !< Sample point.
   integer(I4P)             :: i, j, k !< Counters.

   if (prof == PROFILE_POLYTROPE) then
      kr = PI / A_SPHERE * norm2(xyz)
      r = 0._R_P
      if (kr < PI) then
         r = 1._R_P
         if (kr > 0._R_P) r = sin(kr) / kr
      endif
   else
      r = 0._R_P
      do k=1, SUB
         do j=1, SUB
            do i=1, SUB
               p = xyz + dxyz * ([real(i, R_P), real(j, R_P), real(k, R_P)] - 0.5_R_P) / SUB - 0.5_R_P * dxyz
               if (norm2(p) < A_SPHERE) r = r + 1._R_P
            enddo
         enddo
      enddo
      r = r / SUB**3
   endif
   endfunction rho_sphere

   pure function phi_sphere(xyz, prof) result(p)
   !< The exact potential of the isolated source (G = 1).
   real(R_P),    intent(in) :: xyz(3) !< Point.
   integer(I4P), intent(in) :: prof   !< Source.
   real(R_P)                :: p      !< Potential.
   real(R_P)                :: rr, kk !< Radius, k.
   real(R_P)                :: mass   !< Mass.

   rr = norm2(xyz)
   if (prof == PROFILE_POLYTROPE) then
      kk = PI / A_SPHERE
      mass = 4._R_P * A_SPHERE**3 / PI
      if (rr < A_SPHERE) then
         p = -mass / A_SPHERE
         if (rr > 0._R_P) then
            p = p - FACTOR / kk**2 * sin(kk * rr) / (kk * rr)
         else
            p = p - FACTOR / kk**2
         endif
      else
         p = -mass / rr
      endif
   else
      mass = 4._R_P * PI * A_SPHERE**3 / 3._R_P
      if (rr < A_SPHERE) then
         p = -2._R_P * PI * (A_SPHERE**2 - rr**2 / 3._R_P)
      else
         p = -mass / rr
      endif
   endif
   endfunction phi_sphere

   ! ---------------------------------------------------------------------------------------------------------------------
   ! formatting
   ! ---------------------------------------------------------------------------------------------------------------------
   pure function itoa(i) result(s)
   !< An integer as a trimmed string.
   integer(I4P), intent(in)  :: i !< Value.
   character(:), allocatable :: s !< String.
   character(16)             :: w !< Buffer.

   write(w, '(I0)') i
   s = trim(w)
   endfunction itoa

   pure function fmt(x) result(s)
   !< A real in scientific notation, trimmed (no blank: ADAM's figure scripts read it as one token).
   real(R_P), intent(in)     :: x !< Value.
   character(:), allocatable :: s !< String.
   character(24)             :: w !< Buffer.

   write(w, '(ES12.5E2)') x
   s = trim(adjustl(w))
   endfunction fmt

   pure function fmtf(x) result(s)
   !< A real with three decimals, trimmed.
   real(R_P), intent(in)     :: x !< Value.
   character(:), allocatable :: s !< String.
   character(24)             :: w !< Buffer.

   write(w, '(F8.3)') x
   s = trim(adjustl(w))
   endfunction fmtf
   subroutine new_fft(s)
   !< A spectral solver of the run's backend: the host's, or with --device the cuFFT one (tests-nvf-oac).
   class(elisha_spectral_solver_object), allocatable, intent(inout) :: s !< Solver.

   if (allocated(s)) deallocate(s)
#ifdef _CUFFT
   if (device) then
      allocate(elisha_spectral_solver_cufft_object :: s)
      return
   endif
#endif
   allocate(elisha_spectral_solver_object :: s)
   endsubroutine new_fft

   subroutine new_mg(s)
   !< A hierarchical solver of the run's backend: the host's, or with --device the OpenACC one (its root on cuFFT).
   class(elisha_hierarchical_solver_object), allocatable, intent(inout) :: s !< Solver.

   if (allocated(s)) deallocate(s)
#ifdef _CUFFT
   if (device) then
      allocate(elisha_hierarchical_solver_oac_object :: s)
      return
   endif
#endif
   allocate(elisha_hierarchical_solver_object :: s)
   endsubroutine new_mg

   subroutine fl_setup(fl, mesh, st, m_, isolated, bc, dg_order)
   !< The facade's setup on the run's backend: with --device the cuFFT and OpenACC molds.
   type(elisha_solver_object), intent(inout)         :: fl          !< The facade.
   type(elisha_mesh_object),   intent(inout)         :: mesh        !< Mesh.
   integer(I4P),               intent(out)           :: st          !< Status.
   character(:), allocatable,  intent(out)           :: m_          !< Message.
   logical,                    intent(in),  optional :: isolated    !< Isolated boundaries.
   integer(I4P),               intent(in),  optional :: bc(2,3)     !< Kinds per face.
   integer(I4P),               intent(in),  optional :: dg_order    !< The wide D(G) order.
#ifdef _CUFFT
   type(elisha_spectral_solver_cufft_object)   :: sm !< The spectral mold.
   type(elisha_hierarchical_solver_oac_object) :: hm !< The hierarchical mold.

   if (device) then
      call fl%setup(mesh, c, st, m_, isolated=isolated, bc=bc, dg_order=dg_order, spectral_mold=sm, &
                    hierarchical_mold=hm)
      return
   endif
#endif
   call fl%setup(mesh, c, st, m_, isolated=isolated, bc=bc, dg_order=dg_order)
   endsubroutine fl_setup

#ifdef _CUFFT
   subroutine select_device
   !< --device: each rank on the GPU of its node-local rank modulo the node's GPUs (none: stop).
   integer :: local_comm, local_rank, nd, ierr_ !< Node communicator, node-local rank, GPUs, MPI error status.

   call MPI_COMM_SPLIT_TYPE(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, local_comm, ierr_)
   call MPI_COMM_RANK(local_comm, local_rank, ierr_)
   call MPI_COMM_FREE(local_comm, ierr_)
   nd = acc_get_num_devices(acc_device_nvidia)
   if (nd < 1) then
      print '(A)', 'device run: no NVIDIA GPU visible to this rank'
      error stop 1
   endif
   call acc_set_device_num(mod(local_rank, nd), acc_device_nvidia)
   call acc_init(acc_device_nvidia)
   endsubroutine select_device
#endif
endprogram elisha_sv_mg_mpi
