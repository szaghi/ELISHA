!< ELISHA verification E1a, E2a, E2b, E3a, E3c: the spectral solver's face kinds and the wide D(G) (ADAM's test_flail_bc).
program elisha_sv_bc_mpi
!< ELISHA verification E1a, E2a, E2b, E3a, E3c: the spectral solver's face kinds and the wide D(G) (ADAM's test_flail_bc).
!<
!< **Provenance.** ADAM's `src/tests/poisson/test_flail_bc.F90` (ADAM issue #84: E1a, E2a, E2b, E3a, E3c), ported onto
!< ELISHA: the configurations, the eigenfunctions, the manufactured solutions, the measures, the bounds and the printed
!< lines are ADAM's; the solver is `elisha_spectral_solver_object` through the entry points ADAM's driver exercised,
!< `initialize_blocks` and `solve_blocks` (an explicit block lattice, no mesh: the multigrid's root and PRISM's t0
!< solves use them), and a refusal is the status code ELISHA returns (`elisha_status_name`), not a stop.
!<
!< **The independent operators.** The checks apply the operators themselves, not the solver's: the centred first
!< derivative D1 of order p (the residual of D(G) = D1 D1, the wide eigenvalues in real space) takes its coefficients
!< from GIDEON's generated rows (`gideon_tables`, `FD_D1_S1..S4`, exact rationals), where ADAM's driver typed them in;
!< PRISM's operators (E3a) are GIDEON's centred FD and FV gradient and divergence of order 2 s
!< (`gideon_fd_gradient_c`, `gideon_fd_divergence_c`, `gideon_fv_gradient_c`, `gideon_fv_divergence_c`), the single
!< truth that replaces ADAM's `adam_fdv_operators_library` (GIDEON reproduces it bitwise). GIDEON is a test-only
!< dependency; ELISHA's own stencil (`elisha_dg_stencil`) is never used to check ELISHA.
!<
!< **Legs** (ADAM's assertions):
!<   - `--compact` (E1a): per axis a discrete eigenfunction of the 7-point operator that satisfies the axis's faces, at
!<     the cell centres (P/P `cos(2 pi k x / L)`, D/D `sin(pi k x / L)`, N/N `cos(pi k x / L)`, D/N `sin(pi (k + 1/2)
!<     x / L)`, N/D `cos(pi (k + 1/2) x / L)`, G/G `sin(pi k i / (n + 1))`), their product the source (plus `RHO0`
!<     without a Dirichlet axis, which the solve must drop): `max |phi - phi_h| / max |phi_h|` over the interior and
!<     the ghost layer, and the energy against the exact sum, both at `TOL_DISCRETE`; 8 configurations (6 in 3-D on
!<     4^3 blocks of 4^3 cells, 2 in 2-D);
!<   - `--wide` (E2a): the same with `dg_order` = 2, 4, 6, 8, the eigenvalue computed in real space (D1 applied twice
!<     to the eigenfunction), the ghost layers read back to depth p;
!<   - `--zero` (E2b): zero-ghost faces with D(G) p = 2..8, preconditioned CG on rank 0: a manufactured `u`, its source
!<     `D(G) u / factor` computed here, recovered to `TOL_PCG`, the ghosts returned against the rules, the CG
!<     iterations at most `MAX_PCG_IT`, at p = 4 within one between 16 and 32 cells per axis; and the compact operator
!<     with zero ghosts equal to ghost-Dirichlet, bitwise; the CG stopped at `max_iterations` = 1 above its tolerance
!<     returns `ELISHA_ERR_NOT_CONVERGED` on every rank with `converged` false and the last iterate (finite, ghosts
!<     included), and the same solver with 200 iterations converges (ELISHA notes E1c, decision C; not ADAM's);
!<   - `--prism` (E3a): `div(grad phi) = f` with PRISM's centred FD and FV operators of order 2 s, s = 1..4, the
!<     solution's returned ghosts read (`TOL_PRISM`; `TOL_PCG` with zero-ghost faces);
!<   - `--free` (E3c): free-space ghosts with D(G) p = 2, 4, 6: the ghost slabs against the direct sum
!<     `-factor sum f dV / (4 pi r)` computed here (`TOL_DISCRETE`), `D(G) phi = factor f` with them (`TOL_FREE`), the CG
!<     iterations at most `MAX_PCG_IT`;
!<   - `--order`: the D/D order against the continuum on 16^3, 32^3, 64^3 cells, compact (`MIN_ORDER`) and wide p = 4
!<     (`MIN_ORDER_WIDE`);
!<   - the refusals (every leg flag runs them too): `--refuse-mixed-periodic` (`ELISHA_ERR_BC_PERIODIC_PAIR`),
!<     `--refuse-mixed-ghost` (`ELISHA_ERR_BC_GHOST_PAIR`), `--refuse-wide-isolated` (`ELISHA_ERR_ISOLATED_WIDE`),
!<     `--refuse-zero-mixed` (`ELISHA_ERR_BC_ZERO_GHOST_PAIR`), `--refuse-free-compact` (`ELISHA_ERR_BC_FREE_GHOSTS`).
!< No leg flag runs every leg (the CI run). A refusal flag alone runs only that refusal: exit 0 when the setup returns
!< the expected code, printed by name; 1 otherwise. ADAM's run had to stop (exit 1); ELISHA returns a status.
!< The blocks are dealt round robin to the ranks, as ADAM's; the printed numbers agree across rank counts to round-off.
!<
!< Usage: `mpirun -np N exe/elisha_sv_bc_mpi [--compact] [--wide] [--zero] [--prism] [--free] [--order]`
!<        `mpirun -np N exe/elisha_sv_bc_mpi --refuse-mixed-periodic | --refuse-mixed-ghost | --refuse-wide-isolated |
!<         --refuse-zero-mixed | --refuse-free-compact`
!< `--device` (only in the device build, `tests-nvf-oac`, `exe-oac/`): the cuFFT spectral backend, each rank on the GPU
!< of its node-local rank; the same bounds.
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
use elisha, only : I4P, I8P, R_P, ELISHA_OK, ELISHA_BC_PERIODIC, ELISHA_BC_DIRICHLET, ELISHA_BC_NEUMANN,          &
                   ELISHA_BC_DIRICHLET_GHOST, ELISHA_BC_ZERO_GHOSTS, ELISHA_BC_FREE_GHOSTS,                     &
                   ELISHA_ERR_BC_PERIODIC_PAIR, ELISHA_ERR_BC_GHOST_PAIR, ELISHA_ERR_ISOLATED_WIDE,             &
                   ELISHA_ERR_BC_ZERO_GHOST_PAIR, ELISHA_ERR_BC_FREE_GHOSTS, elisha_status_name,                &
                   ELISHA_ERR_NOT_CONVERGED,                                                                    &
                   elisha_comm_object, elisha_spectral_solver_object
use gideon_tables, only : FD_D1_S1, FD_D1_S2, FD_D1_S3, FD_D1_S4
use gideon_operators_3d, only : gideon_fd_gradient_c, gideon_fd_divergence_c, gideon_fv_gradient_c, &
                                gideon_fv_divergence_c
implicit none
real(R_P),    parameter :: PI=acos(-1._R_P)              !< Pi.
real(R_P),    parameter :: FACTOR=4._R_P * PI            !< Source factor.
real(R_P),    parameter :: RHO0=1.5_R_P                  !< Constant added without a Dirichlet axis (dropped).
real(R_P),    parameter :: L(3)=[1._R_P, 2._R_P, 3._R_P] !< Domain lengths.
real(R_P),    parameter :: TOL_DISCRETE=1.e-12_R_P       !< Bound on the discrete error and the energy, relative.
real(R_P),    parameter :: MIN_ORDER=1.9_R_P             !< Bound on the D/D order against the continuum.
real(R_P),    parameter :: MIN_ORDER_WIDE=3.8_R_P        !< Bound on the D/D order of the wide p = 4 operator.
real(R_P),    parameter :: TOL_PCG=1.e-9_R_P             !< Bound on the zero-ghost solution's error, relative.
integer(I4P), parameter :: MAX_PCG_IT=40_I4P             !< Bound on the zero-ghost CG iterations.
real(R_P),    parameter :: TOL_PRISM=1.e-11_R_P          !< Bound on |div(grad phi) - f| / max |f|, direct solves.
real(R_P),    parameter :: TOL_FREE=1.e-10_R_P           !< Bound on |D(G) phi - factor f| / max |factor f|, free ghosts.
integer(I4P), parameter :: P=ELISHA_BC_PERIODIC          !< Shorthand.
integer(I4P), parameter :: D=ELISHA_BC_DIRICHLET         !< Shorthand.
integer(I4P), parameter :: N=ELISHA_BC_NEUMANN           !< Shorthand.
integer(I4P), parameter :: G=ELISHA_BC_DIRICHLET_GHOST   !< Shorthand.
integer(I4P), parameter :: Z=ELISHA_BC_ZERO_GHOSTS       !< Shorthand.
integer(I4P), parameter :: F=ELISHA_BC_FREE_GHOSTS       !< Shorthand.
integer(I4P), parameter :: MAX_ROWS=64_I4P               !< Rows of the summary table.
! named actual arguments: ifx's -check arg_temp_created reports every array expression passed to an explicit-shape
! dummy (a warning and a traceback per call), so constructors and expressions go through these
logical,      parameter :: NO_NULL(3)=[.false., .false., .false.] !< No null axis.
integer(I4P), parameter :: B111(3)=[1_I4P, 1_I4P, 1_I4P]           !< One block.
integer(I4P), parameter :: B222(3)=[2_I4P, 2_I4P, 2_I4P]           !< 2^3 blocks.
integer(I4P), parameter :: C444(3)=[4_I4P, 4_I4P, 4_I4P]           !< 4^3 cells.
integer(I4P), parameter :: C888(3)=[8_I4P, 8_I4P, 8_I4P]           !< 8^3 cells.
real(R_P),    parameter :: DX01(3)=[0.1_R_P, 0.1_R_P, 0.1_R_P]     !< A cubic spacing.
real(R_P),    parameter :: DXL16(3)=L / 16._R_P                    !< SV-2's box over 16 cells.
real(R_P),    parameter :: DXL8(3)=L / 8._R_P                      !< SV-2's box over 8 cells.
type(elisha_comm_object)  :: c             !< Communication context (the world).
real(R_P)                 :: a_d1(4,4)     !< Centred D1 coefficients a_j of order p, column p/2 (GIDEON's rows).
logical                   :: test_passed   !< Aggregate pass flag.
logical                   :: leg(6)        !< Legs: compact, wide, zero, prism, free, order.
character(64)             :: refusal       !< The only refusal to run ('': none).
character(256)            :: arg           !< Command argument.
character(320)            :: rows(MAX_ROWS) !< Summary rows.
integer(I4P)              :: nrows         !< Summary rows used.
integer(I4P)              :: cfg_bc(2,3)   !< Current case: kinds per face.
integer(I4P)              :: cfg_kmode(3)  !< Current case: mode per axis.
integer(I4P)              :: cfg_ncell(3)  !< Current case: global cells per axis.
integer(I4P)              :: cfg_order     !< Current case: 0 compact, p the wide D(G).
logical                   :: cfg_null(3)   !< Current case: null directions.
logical                   :: cfg_dirichlet !< Current case: some axis has a Dirichlet face.
integer(I4P)              :: a, po, m, stat !< Counters, status.
real(R_P)                 :: lmax(2)       !< A leg's maxima of its two measures.
integer(I4P)              :: bcx(2,3)      !< Kinds per face, a named actual argument.
integer(I4P)              :: n3a(3), n3b(3) !< Blocks and cells per axis, named actual arguments.
integer(I4P)              :: lit(2)        !< A leg's CG iteration range.
logical                   :: lok           !< A leg's verdict.
integer                   :: ierr, world   !< MPI error status, the world communicator.
logical                   :: halting(3)    !< The FP halting modes of the build (debug: on).
logical                   :: device                      !< --device: the device backends.
real(R_P)                 :: t0, tl        !< Wall clock at the start, at a leg's start.
character(:), allocatable :: msg           !< Message.

a_d1 = 0._R_P
a_d1(1:1,1) = FD_D1_S1 ; a_d1(1:2,2) = FD_D1_S2 ; a_d1(1:3,3) = FD_D1_S3 ; a_d1(1:4,4) = FD_D1_S4
leg = .false. ; refusal = '' ; nrows = 0 ; cfg_order = 0
device = .false.
a = 0
do while (a < command_argument_count())
   a = a + 1
   call get_command_argument(a, arg)
   select case(trim(arg))
   case('--compact') ; leg(1) = .true.
   case('--wide')    ; leg(2) = .true.
   case('--zero')    ; leg(3) = .true.
   case('--prism')   ; leg(4) = .true.
   case('--free')    ; leg(5) = .true.
   case('--order')   ; leg(6) = .true.
   case('--device')   ; device = .true.
   case default
      if (index(arg, '--refuse-') == 1) then
         refusal = arg(10:)
      else
         print '(A)', 'elisha_sv_bc_mpi: unknown argument "'//trim(arg)//'"'
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
   print '(A)', 'elisha_sv_bc_mpi: --device needs the device build (tests-nvf-oac)'
   error stop 2
endif
#endif
call c%init(world, stat, msg)
if (stat /= ELISHA_OK) then
   print '(A)', 'elisha_sv_bc_mpi: comm init: '//msg
   error stop 1
endif
t0 = c%wtime()
test_passed = .true.
if (len_trim(refusal) > 0) then
   select case(trim(refusal))
   case('mixed-periodic', 'mixed-ghost', 'wide-isolated', 'zero-mixed', 'free-compact')
      call check_refusal(trim(refusal))
   case default
      if (c%rank == 0) print '(A)', 'elisha_sv_bc_mpi: unknown refusal "'//trim(refusal)//'"'
      call finish
      error stop 2
   endselect
else
   ! the compact operator (E1a) and the wide D(G) of order 2, 4, 6, 8 (E2a)
   do po=0, 8, 2
      if (po == 0 .and. .not.leg(1)) cycle
      if (po > 0 .and. .not.leg(2)) cycle
      tl = c%wtime()
      if (po == 0 .and. c%rank == 0) print '(A)', '   the compact operator (E1a):'
      if (po == 2 .and. c%rank == 0) print '(A)', '   the wide operator D(G) (E2a):'
      cfg_order = po
      call leg_start
      bcx = reshape([P,P, D,D, N,N], [2,3])
      call check('P/P D/D N/N', bcx, null_z=.false.)
      bcx = reshape([D,N, N,D, P,P], [2,3])
      call check('D/N N/D P/P', bcx, null_z=.false.)
      bcx = reshape([N,N, N,N, N,N], [2,3])
      call check('N/N N/N N/N', bcx, null_z=.false.)
      bcx = reshape([D,D, D,D, D,D], [2,3])
      call check('D/D D/D D/D', bcx, null_z=.false.)
      bcx = reshape([N,D, D,N, N,N], [2,3])
      call check('N/D D/N N/N', bcx, null_z=.false.)
      bcx = reshape([G,G, G,G, G,G], [2,3])
      call check('G/G G/G G/G', bcx, null_z=.false.)
      bcx = reshape([D,N, N,N, P,P], [2,3])
      call check('2-D D/N N/N ', bcx, null_z=.true.)
      bcx = reshape([P,P, D,D, P,P], [2,3])
      call check('2-D P/P D/D ', bcx, null_z=.true.)
      call leg_row(merge('E1a compact      ', 'E2a D(G) p = '//itoa(po)//'   ', po == 0), '8 configurations, '// &
                   '|phi - phi_h| max '//fmt(lmax(1))//', energy max '//fmt(lmax(2))//' (<= '//fmt(TOL_DISCRETE)//')')
   enddo
   if (leg(1)) then
      call check_refusal('mixed-periodic')
      call check_refusal('mixed-ghost')
   endif
   if (leg(2)) call check_refusal('wide-isolated')
   ! zero-ghost faces (E2b): preconditioned CG, a manufactured solution
   if (leg(3)) then
      if (c%rank == 0) print '(A)', '   zero-ghost faces (E2b), preconditioned CG:'
      do po=2, 8, 2
         tl = c%wtime()
         call leg_start
         bcx = reshape([Z,Z, P,P, P,P], [2,3])
         n3a = [4,4,4] ; n3b = [4,4,4]
         call check_zero('Z/Z P/P P/P', bcx, po, n3a, n3b, .false., m)
         bcx = reshape([Z,Z, D,D, N,N], [2,3])
         n3a = [4,4,4] ; n3b = [4,4,4]
         call check_zero('Z/Z D/D N/N', bcx, po, n3a, n3b, .false., m)
         bcx = reshape([Z,Z, Z,Z, Z,Z], [2,3])
         n3a = [4,4,4] ; n3b = [4,4,4]
         call check_zero('Z/Z Z/Z Z/Z', bcx, po, n3a, n3b, .false., m)
         bcx = reshape([Z,Z, N,N, P,P], [2,3])
         n3a = [4,4,1] ; n3b = [4,4,1]
         call check_zero('2-D Z/Z N/N', bcx, po, n3a, n3b, .true.,  m)
         call leg_row('E2b D(G) p = '//itoa(po)//'   ', '4 configurations, |u - u_exact| max '//fmt(lmax(1))//         &
                      ' (<= '//fmt(TOL_PCG)//'), ghosts max '//fmt(lmax(2))//', CG iterations '//itoa(lit(1))//'-'// &
                      itoa(lit(2))//' (<= '//itoa(MAX_PCG_IT)//')')
      enddo
      block
      integer(I4P) :: it16, it32 !< Iterations on 16 and 32 cells per axis.
      logical      :: ok         !< Pass flag.
      tl = c%wtime()
      call leg_start
      bcx = reshape([Z,Z, D,D, N,N], [2,3])
      n3a = [4,4,4] ; n3b = [4,4,4]
      call check_zero('Z/Z D/D N/N', bcx, 4_I4P, n3a, n3b, .false., it16)
      bcx = reshape([Z,Z, D,D, N,N], [2,3])
      n3a = [4,4,4] ; n3b = [8,8,8]
      call check_zero('Z/Z D/D N/N', bcx, 4_I4P, n3a, n3b, .false., it32)
      ok = abs(it32 - it16) <= 1
      if (c%rank == 0) print '(A)', '   p = 4, Z/Z D/D N/N, CG iterations on 16 and 32 cells per axis: '// &
         itoa(it16)//', '//itoa(it32)//merge(' ok  ', ' FAIL', ok)
      lok = lok .and. ok
      call leg_row('E2b grid bound   ', 'p = 4 Z/Z D/D N/N, CG iterations 16^3 '//itoa(it16)//', 32^3 '//itoa(it32)// &
                   ' (within 1)')
      endblock
      tl = c%wtime()
      call leg_start
      call check_zero_compact
      call leg_row('E2b compact Z=G  ', 'Z/Z D/N P/P against G/G D/N P/P, max |diff| '//fmt(lmax(1))//' (bitwise)')
      tl = c%wtime()
      call leg_start
      call check_zero_not_converged
      call leg_row('E2b not converged', 'p = 4 Z/Z D/D N/N, max_iterations 1: ELISHA_ERR_NOT_CONVERGED, the last '// &
                   'iterate finite; 200: converged')
      call check_refusal('zero-mixed')
   endif
   ! PRISM's own operators (E3a): div(grad phi) = f with GIDEON's FD and FV centred derivatives
   if (leg(4)) then
      if (c%rank == 0) print '(A)', '   PRISM''s operators (E3a), div(grad phi) - f with GIDEON''s centred FD and FV:'
      do po=1, 4
         tl = c%wtime()
         call leg_start
         bcx = reshape([P,P, P,P, P,P], [2,3])
         call check_prism_ops('P/P P/P P/P', bcx, po)
         bcx = reshape([D,D, D,D, D,D], [2,3])
         call check_prism_ops('D/D D/D D/D', bcx, po)
         bcx = reshape([N,N, D,D, D,D], [2,3])
         call check_prism_ops('N/N D/D D/D', bcx, po)
         bcx = reshape([D,N, P,P, N,D], [2,3])
         call check_prism_ops('D/N P/P N/D', bcx, po)
         bcx = reshape([Z,Z, Z,Z, Z,Z], [2,3])
         call check_prism_ops('Z/Z Z/Z Z/Z', bcx, po)
         bcx = reshape([Z,Z, D,D, N,N], [2,3])
         call check_prism_ops('Z/Z D/D N/N', bcx, po)
         call leg_row('E3a order '//itoa(2*po)//'      ', '6 configurations, |div(grad phi) - f| / max|f| FD max '// &
                      fmt(lmax(1))//', FV max '//fmt(lmax(2))//' (<= '//fmt(TOL_PRISM)//', zero ghosts '//      &
                      fmt(TOL_PCG)//')')
      enddo
   endif
   ! free-space ghosts (E3c)
   if (leg(5)) then
      if (c%rank == 0) print '(A)', '   free-space ghosts (E3c), preconditioned CG:'
      do po=2, 6, 2
         tl = c%wtime()
         call leg_start
         bcx = reshape([F,F, F,F, F,F], [2,3])
         call check_free('F/F F/F F/F', bcx, po)
         bcx = reshape([F,F, D,D, P,P], [2,3])
         call check_free('F/F D/D P/P', bcx, po)
         bcx = reshape([F,Z, N,N, F,F], [2,3])
         call check_free('F/Z N/N F/F', bcx, po)
         call leg_row('E3c D(G) p = '//itoa(po)//'   ', '3 configurations, ghosts against the direct sum max '//      &
                      fmt(lmax(1))//' (<= '//fmt(TOL_DISCRETE)//'), |D(G) phi - factor f| max '//fmt(lmax(2))//   &
                      ' (<= '//fmt(TOL_FREE)//'), CG iterations '//itoa(lit(1))//'-'//itoa(lit(2))//' (<= '//     &
                      itoa(MAX_PCG_IT)//')')
      enddo
      call check_refusal('free-compact')
   endif
   ! the D/D orders against the continuum: wide p = 4 and compact
   if (leg(6)) then
      call check_order(4_I4P, MIN_ORDER_WIDE, 'wide p = 4, D/D D/D D/D order against the continuum, 16^3 -> 32^3 -> 64^3')
      call check_order(0_I4P, MIN_ORDER, 'D/D D/D D/D order against the continuum, 16^3 -> 32^3 -> 64^3 cells per axis')
   endif
endif
if (c%rank == 0) then
   print '(A,I0,A,F8.2,A)', 'elisha_sv_bc_mpi summary (np ', c%nranks, ', wall ', c%wtime() - t0, ' s):'
   do a=1, nrows
      print '(A)', '  '//trim(rows(a))
   enddo
   if (test_passed) then
      print '(A)', 'elisha_sv_bc_mpi: PASSED'
   else
      print '(A)', 'elisha_sv_bc_mpi: FAILED'
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

   ! ---------------------------------------------------------------------------------------------------------------------
   ! the summary: per leg row, the maxima and the iteration range of its checks
   ! ---------------------------------------------------------------------------------------------------------------------
   subroutine leg_start
   !< Reset the leg accumulators.
   lmax = 0._R_P ; lit = [huge(1_I4P), 0_I4P] ; lok = .true.
   endsubroutine leg_start

   subroutine leg_add(e1, e2, ok, its)
   !< Fold one check into the leg accumulators and the verdict.
   real(R_P),    intent(in)           :: e1, e2 !< The two measures.
   logical,      intent(in)           :: ok     !< Pass flag.
   integer(I4P), intent(in), optional :: its    !< CG iterations.

   lmax = max(lmax, [e1, e2])
   lok = lok .and. ok
   if (present(its)) lit = [min(lit(1), its), max(lit(2), its)]
   test_passed = test_passed .and. ok
   endsubroutine leg_add

   subroutine leg_row(name, what)
   !< A summary row for the leg, with its verdict and wall time.
   character(*), intent(in) :: name !< Leg.
   character(*), intent(in) :: what !< The measures with their bounds.
   character(16)            :: w    !< Buffer.

   test_passed = test_passed .and. lok
   write(w, '(F10.2)') c%wtime() - tl
   if (nrows >= MAX_ROWS) return
   nrows = nrows + 1
   rows(nrows) = name//' '//what//' | '//merge('PASS', 'FAIL', lok)//' | '//trim(adjustl(w))//' s'
   endsubroutine leg_row

   ! ---------------------------------------------------------------------------------------------------------------------
   ! the legs
   ! ---------------------------------------------------------------------------------------------------------------------
   subroutine check(label, bc, null_z)
   !< E1a, E2a: one configuration on 4^3 blocks of 4^3 cells (2-D: 4^2 blocks of 4^2); the discrete error and the energy.
   character(*), intent(in) :: label   !< Configuration label.
   integer(I4P), intent(in) :: bc(2,3) !< Kinds per face [lo/hi, axis].
   logical,      intent(in) :: null_z  !< 2-D.
   integer(I4P)             :: nblk(3), nijk(3) !< Blocks and cells per block per direction.
   real(R_P)                :: e_disc, e_cont, e_energy !< Errors.
   logical                  :: ok      !< Pass flag.

   nblk = [4,4,4] ; nijk = [4,4,4]
   if (null_z) then
      nblk(3) = 1 ; nijk(3) = 1
   endif
   call solve_case(bc, nblk, nijk, null_z, e_disc, e_cont, e_energy)
   ok = e_disc <= TOL_DISCRETE .and. e_energy <= TOL_DISCRETE
   call leg_add(e_disc, e_energy, ok)
   if (c%rank == 0) print '(A)', '   '//op_label()//' '//label//': |phi - phi_h| '//fmt(e_disc)//', energy '// &
      fmt(e_energy)//merge(' ok  ', ' FAIL', ok)
   endsubroutine check

   subroutine check_order(order_, min_ord, label)
   !< The D/D order against the continuum on 16^3, 32^3 and 64^3 cells (2^3 blocks), the finest pair asserted.
   integer(I4P), intent(in) :: order_  !< 0 compact, p the wide D(G).
   real(R_P),    intent(in) :: min_ord !< Bound.
   character(*), intent(in) :: label   !< Line label.
   real(R_P)                :: e_disc, e_cont(3), e_energy, order !< Errors, order.
   integer(I4P)             :: mm      !< Counter.
   logical                  :: ok      !< Pass flag.

   tl = c%wtime()
   cfg_order = order_
   do mm=1, 3
      bcx = reshape([D,D, D,D, D,D], [2,3])
      n3b = [4,4,4]*2**mm
      call solve_case(bcx, B222, n3b, .false., e_disc, e_cont(mm), e_energy)
   enddo
   order = log(e_cont(2) / e_cont(3)) / log(2._R_P)
   ok = order >= min_ord
   if (c%rank == 0) print '(A)', '   '//label//': '//fmtf(log(e_cont(1) / e_cont(2)) / log(2._R_P))//', '// &
      fmtf(order)//' (min '//fmtf(min_ord)//')'//merge(' ok  ', ' FAIL', ok)
   lok = ok
   call leg_row(merge('order compact    ', 'order D(G) p = 4 ', order_ == 0), 'D/D D/D D/D against the continuum, '// &
                'errors '//fmt(e_cont(1))//', '//fmt(e_cont(2))//', '//fmt(e_cont(3))//', orders '//                 &
                fmtf(log(e_cont(1) / e_cont(2)) / log(2._R_P))//', '//fmtf(order)//' (>= '//fmtf(min_ord)//')')
   endsubroutine check_order

   subroutine solve_case(bc, nblk, nijk, null_z, e_disc, e_cont, e_energy)
   !< Build the source of a configuration on this rank's blocks (round robin), solve, and return the errors: discrete
   !< (interior and the ghost layers read), continuum (interior: the mode over -|k|^2) and energy.
   integer(I4P), intent(in)  :: bc(2,3)   !< Kinds per face [lo/hi, axis].
   integer(I4P), intent(in)  :: nblk(3)   !< Blocks per direction.
   integer(I4P), intent(in)  :: nijk(3)   !< Cells per block per direction.
   logical,      intent(in)  :: null_z    !< 2-D.
   real(R_P),    intent(out) :: e_disc    !< Discrete error, relative.
   real(R_P),    intent(out) :: e_cont    !< Continuum error, max.
   real(R_P),    intent(out) :: e_energy  !< Energy error, relative.
   class(elisha_spectral_solver_object), allocatable :: solver !< The solver.
   integer(I4P)              :: gw         !< Ghost layers read back: 1, or p for the wide operator.
   integer(I4P)              :: ncell(3)   !< Global cells per direction.
   real(R_P)                 :: dxyz(3)    !< Spacing.
   real(R_P)                 :: mu(3)      !< Discrete eigenvalue per axis.
   real(R_P)                 :: kap(3)     !< Continuum wavenumber per axis.
   integer(I4P)              :: kmode(3)   !< Mode per axis.
   logical                   :: null_xyz(3) !< Null directions.
   integer(I4P), allocatable :: coords(:,:) !< This rank's block coordinates.
   real(R_P),    allocatable :: buf(:), out(:) !< Packed source and potential.
   real(R_P)                 :: ph, ph_c, e, red(3), esum(1) !< Work.
   integer(I4P)              :: nbl, b, i, j, k, cc, d_ !< Counters.

   call new_fft(solver)
   e_disc = huge(1._R_P) ; e_cont = huge(1._R_P) ; e_energy = huge(1._R_P)
   gw = max(1_I4P, cfg_order)
   null_xyz = [.false., .false., null_z]
   ncell = nblk * nijk
   dxyz = L / ncell
   kmode = [2, 1, 1]
   cfg_bc = bc ; cfg_kmode = kmode ; cfg_ncell = ncell ; cfg_null = null_xyz
   cfg_dirichlet = any(bc == D) .or. any(bc == G)
   do d_=1, 3
      call axis_eigen(bc(:,d_), kmode(d_), ncell(d_), dxyz(d_), null_xyz(d_), cfg_order, mu(d_), kap(d_))
   enddo
   call round_robin(nblk, coords, nbl)
   allocate(buf(max(nbl*product(nijk),1)), out(max(nbl*product(nijk+2*gw),1)))
   buf = 0._R_P
   cc = 0
   do b=1, nbl
      do k=1, nijk(3)
         do j=1, nijk(2)
            do i=1, nijk(1)
               cc = cc + 1
               buf(cc) = source(coords(1,b)*nijk(1)+i, coords(2,b)*nijk(2)+j, coords(3,b)*nijk(3)+k)
            enddo
         enddo
      enddo
   enddo
   call solver%initialize_blocks(nblk=nblk, nijk=nijk, ngc=gw, dxyz=dxyz, level=0_I4P, null_xyz=null_xyz, &
                                 isolated=.false., comm=c, stat=stat, errmsg=msg, bc=bc, dg_order=cfg_order)
   if (failed('initialize_blocks')) return
   out = 0._R_P
   call solver%solve_blocks(nbl=nbl, coords=coords, buf=buf, factor=FACTOR, gw=gw, out=out, stat=stat, errmsg=msg, &
                            energy=e)
   if (failed('solve_blocks')) return
   red = 0._R_P ; esum = 0._R_P
   cc = 0
   do b=1, nbl
      do k=1-gw, nijk(3)+gw ! along a null z the solve stores the ghost layers too; the mode is 1 there
         do j=1-gw, nijk(2)+gw
            do i=1-gw, nijk(1)+gw
               cc = cc + 1
               ph = FACTOR * mode(coords(1,b)*nijk(1)+i, coords(2,b)*nijk(2)+j, coords(3,b)*nijk(3)+k) / sum(mu)
               red(1) = max(red(1), abs(out(cc) - ph))
               red(2) = max(red(2), abs(ph))
               if (i>=1 .and. i<=nijk(1) .and. j>=1 .and. j<=nijk(2) .and. k>=1 .and. k<=nijk(3)) then
                  esum(1) = esum(1) + source(coords(1,b)*nijk(1)+i, coords(2,b)*nijk(2)+j, coords(3,b)*nijk(3)+k) * ph
                  ph_c = -FACTOR * mode(coords(1,b)*nijk(1)+i, coords(2,b)*nijk(2)+j, coords(3,b)*nijk(3)+k) / &
                         sum(kap**2)
                  red(3) = max(red(3), abs(out(cc) - ph_c))
               endif
            enddo
         enddo
      enddo
   enddo
   call c%max_real8(red, stat)
   call c%sum_r(esum, stat)
   e_disc = red(1) / red(2)
   e_cont = red(3)
   e_energy = abs(e - esum(1)) / abs(esum(1))
   call solver%destroy
   endsubroutine solve_case

   subroutine check_zero(label, bc, p_, nblk, nijk, null_z, iters)
   !< E2b: zero-ghost faces with the wide D(G) of order p_: the manufactured `u`, its source computed here, solved; the
   !< error, the ghosts returned against the rules, the CG iterations asserted.
   character(*), intent(in)  :: label   !< Configuration label.
   integer(I4P), intent(in)  :: bc(2,3) !< Kinds per face [lo/hi, axis].
   integer(I4P), intent(in)  :: p_      !< Order.
   integer(I4P), intent(in)  :: nblk(3), nijk(3) !< Blocks and cells per block per direction.
   logical,      intent(in)  :: null_z  !< 2-D.
   integer(I4P), intent(out) :: iters   !< CG iterations.
   class(elisha_spectral_solver_object), allocatable :: solver !< The solver.
   integer(I4P)              :: ncell(3), nbl, b, i, j, k, cc, it(1) !< Sizes, counters.
   real(R_P)                 :: dxyz(3)  !< Spacing.
   logical                   :: null_xyz(3) !< Null directions.
   real(R_P),    allocatable :: u(:,:,:), f(:,:,:), uo(:,:,:), uflat(:) !< Exact solution, source, solution returned.
   integer(I4P), allocatable :: coords(:,:) !< This rank's block coordinates.
   real(R_P),    allocatable :: buf(:), out(:) !< Packed source and potential.
   real(R_P)                 :: emax, umax, gm(1) !< Errors.
   logical                   :: ok  !< Pass flag.

   call new_fft(solver)
   iters = huge(1_I4P)
   null_xyz = [.false., .false., null_z]
   ncell = nblk * nijk
   dxyz = L / ncell
   allocate(u(1-p_:ncell(1)+p_,1-p_:ncell(2)+p_,1-p_:ncell(3)+p_), f(ncell(1),ncell(2),ncell(3)))
   allocate(uo, mold=u)
   u = 0._R_P
   do k=1, ncell(3)
      do j=1, ncell(2)
         do i=1, ncell(1)
            u(i,j,k) = sin(0.7_R_P * i + 0.3_R_P) * cos(0.5_R_P * j - 0.2_R_P) * cos(0.4_R_P * k + 0.1_R_P) + &
                       0.25_R_P * cos(1.1_R_P * i - 0.9_R_P * j + 0.3_R_P * k)
            if (null_z) u(i,j,k) = sin(0.7_R_P * i + 0.3_R_P) * cos(0.5_R_P * j - 0.2_R_P) + &
                                   0.25_R_P * cos(1.1_R_P * i - 0.9_R_P * j)
         enddo
      enddo
   enddo
   call rule_ghosts(u, ncell, bc, null_xyz, p_)
   call d1d1(u, ncell, dxyz, null_xyz, p_, f)
   f = f / FACTOR
   call round_robin(nblk, coords, nbl)
   allocate(buf(max(nbl*product(nijk),1)), out(max(nbl*product(nijk+2*p_),1)))
   buf = 0._R_P
   cc = 0
   do b=1, nbl
      do k=1, nijk(3)
         do j=1, nijk(2)
            do i=1, nijk(1)
               cc = cc + 1
               buf(cc) = f(coords(1,b)*nijk(1)+i, coords(2,b)*nijk(2)+j, coords(3,b)*nijk(3)+k)
            enddo
         enddo
      enddo
   enddo
   call solver%initialize_blocks(nblk=nblk, nijk=nijk, ngc=p_, dxyz=dxyz, level=0_I4P, null_xyz=null_xyz, &
                                 isolated=.false., comm=c, stat=stat, errmsg=msg, bc=bc, dg_order=p_)
   if (failed('initialize_blocks')) return
   out = 0._R_P
   call solver%solve_blocks(nbl=nbl, coords=coords, buf=buf, factor=FACTOR, gw=p_, out=out, stat=stat, errmsg=msg)
   if (failed('solve_blocks')) return
   it = solver%iterations ! rank 0 runs the CG
   call c%max_i4(it, stat)
   iters = it(1)
   ! the interior returned, then the rules applied to it, against the ghosts returned
   uo = 0._R_P
   cc = 0
   do b=1, nbl
      do k=1-p_, nijk(3)+p_
         do j=1-p_, nijk(2)+p_
            do i=1-p_, nijk(1)+p_
               cc = cc + 1
               if (i>=1 .and. i<=nijk(1) .and. j>=1 .and. j<=nijk(2) .and. k>=1 .and. k<=nijk(3)) &
                  uo(coords(1,b)*nijk(1)+i, coords(2,b)*nijk(2)+j, coords(3,b)*nijk(3)+k) = out(cc)
            enddo
         enddo
      enddo
   enddo
   uflat = reshape(uo, [size(uo)])
   call c%sum_r(uflat, stat) ! each cell from one rank
   uo = reshape(uflat, shape(uo))
   emax = maxval(abs(uo(1:ncell(1),1:ncell(2),1:ncell(3)) - u(1:ncell(1),1:ncell(2),1:ncell(3))))
   umax = maxval(abs(u(1:ncell(1),1:ncell(2),1:ncell(3))))
   call rule_ghosts(uo, ncell, bc, null_xyz, p_)
   gm = 0._R_P
   cc = 0
   do b=1, nbl
      do k=1-p_, nijk(3)+p_
         do j=1-p_, nijk(2)+p_
            do i=1-p_, nijk(1)+p_
               cc = cc + 1
               if (null_z .and. (k < 1 .or. k > nijk(3))) cycle ! a null z: the ghost layers copy the cell
               gm(1) = max(gm(1), abs(out(cc) - uo(coords(1,b)*nijk(1)+i, coords(2,b)*nijk(2)+j, coords(3,b)*nijk(3)+k)))
            enddo
         enddo
      enddo
   enddo
   call c%max_real8(gm, stat)
   ok = emax / umax <= TOL_PCG .and. gm(1) <= TOL_DISCRETE * umax .and. iters <= MAX_PCG_IT
   call leg_add(emax / umax, gm(1), ok, iters)
   if (c%rank == 0) print '(A)', '   D(G) p = '//itoa(p_)//' '//label//' ('//itoa(ncell(1))//' cells per axis): '// &
      '|u - u_exact| '//fmt(emax / umax)//', ghosts against the rules '//fmt(gm(1))//', CG iterations '//itoa(iters)// &
      merge(' ok  ', ' FAIL', ok)
   call solver%destroy
   endsubroutine check_zero

   subroutine check_free(label, bc, p_)
   !< E3c: free-space ghosts with the wide D(G) of order p_: a smooth source on 2^3 blocks of 4^3 cells, solved; the
   !< ghost slabs of the free faces against the direct sum `-FACTOR sum_y f(y) dV / (4 pi |x - y|)` computed here, and
   !< D(G) of the solution (its ghosts: the slabs returned on the free faces, the test's rules on the others) against
   !< `FACTOR f`.
   character(*), intent(in)  :: label   !< Configuration label.
   integer(I4P), intent(in)  :: bc(2,3) !< Kinds per face [lo/hi, axis].
   integer(I4P), intent(in)  :: p_      !< Order.
   class(elisha_spectral_solver_object), allocatable :: solver !< The solver.
   integer(I4P), parameter   :: NBLK(3)=[2,2,2], NIJK(3)=[4,4,4] !< Blocks and cells per block per direction.
   integer(I4P)              :: ncell(3), nbl, gw, b, i, j, k, ii, jj, kk, cc, ax, side !< Counters.
   integer(I4P)              :: lo(3), hi(3), x(3), it(1) !< A slab, a global cell, CG iterations.
   integer(I4P)              :: bc_rule(2,3) !< The kinds with the free faces zeroed (the test's rules).
   real(R_P)                 :: dxyz(3), s, gref, gmax, gscale, res !< Spacing, sums, errors.
   real(R_P),    allocatable :: src(:,:,:), uo(:,:,:), y(:,:,:), ua(:,:,:), uflat(:) !< Source, solution, D(G) of it.
   integer(I4P), allocatable :: coords(:,:) !< This rank's block coordinates.
   real(R_P),    allocatable :: buf(:), out(:) !< Packed source and potential.
   logical                   :: ok  !< Pass flag.

   call new_fft(solver)
   gw = max(p_, 1_I4P)
   ncell = NBLK * NIJK
   dxyz = L / ncell
   allocate(src(ncell(1),ncell(2),ncell(3)), uo(1-gw:ncell(1)+gw,1-gw:ncell(2)+gw,1-gw:ncell(3)+gw))
   allocate(y(ncell(1),ncell(2),ncell(3)))
   do k=1, ncell(3)
      do j=1, ncell(2)
         do i=1, ncell(1)
            src(i,j,k) = sin(0.7_R_P * i + 0.3_R_P) * cos(0.5_R_P * j - 0.2_R_P) * cos(0.4_R_P * k + 0.1_R_P) + 0.3_R_P
         enddo
      enddo
   enddo
   call round_robin(NBLK, coords, nbl)
   allocate(buf(max(nbl*product(NIJK),1)), out(max(nbl*product(NIJK+2*gw),1)))
   buf = 0._R_P
   cc = 0
   do b=1, nbl
      do k=1, NIJK(3)
         do j=1, NIJK(2)
            do i=1, NIJK(1)
               cc = cc + 1
               buf(cc) = src(coords(1,b)*NIJK(1)+i, coords(2,b)*NIJK(2)+j, coords(3,b)*NIJK(3)+k)
            enddo
         enddo
      enddo
   enddo
   call solver%initialize_blocks(nblk=NBLK, nijk=NIJK, ngc=gw, dxyz=dxyz, level=0_I4P,                       &
                                 null_xyz=NO_NULL, isolated=.false., comm=c, stat=stat, &
                                 errmsg=msg, bc=bc, dg_order=p_)
   if (failed('initialize_blocks')) return
   out = 0._R_P
   call solver%solve_blocks(nbl=nbl, coords=coords, buf=buf, factor=FACTOR, gw=gw, out=out, stat=stat, errmsg=msg)
   if (failed('solve_blocks')) return
   it = solver%iterations
   call c%max_i4(it, stat)
   ! the interior and the face ghost slabs returned (each cell from one block), on every rank
   uo = 0._R_P
   cc = 0
   do b=1, nbl
      do k=1-gw, NIJK(3)+gw
         do j=1-gw, NIJK(2)+gw
            do i=1-gw, NIJK(1)+gw
               cc = cc + 1
               x = coords(:,b) * NIJK + [i, j, k]
               ! a block's own cells, and the face-slab cells outside the domain along one axis and inside this block
               ! along the other two, from the boundary block of that axis only: edges, corners and inter-block ghosts
               ! are taken from no block
               if (count(x < 1 .or. x > ncell) > 1) cycle
               if (count([i, j, k] < 1 .or. [i, j, k] > NIJK) /= count(x < 1 .or. x > ncell)) cycle
               if (any((x < 1 .and. coords(:,b) /= 0) .or. (x > ncell .and. coords(:,b) /= NBLK - 1))) cycle
               uo(x(1),x(2),x(3)) = out(cc)
            enddo
         enddo
      enddo
   enddo
   uflat = reshape(uo, [size(uo)])
   call c%sum_r(uflat, stat)
   uo = reshape(uflat, shape(uo))
   ! the free slabs against the direct sum
   gmax = 0._R_P ; gscale = 0._R_P
   do ax=1, 3
      do side=1, 2
         if (bc(side,ax) /= F) cycle
         lo = 1 ; hi = ncell
         if (side == 1) then
            lo(ax) = 1 - gw ; hi(ax) = 0
         else
            lo(ax) = ncell(ax) + 1 ; hi(ax) = ncell(ax) + gw
         endif
         do k=lo(3), hi(3)
            do j=lo(2), hi(2)
               do i=lo(1), hi(1)
                  s = 0._R_P
                  do kk=1, ncell(3)
                     do jj=1, ncell(2)
                        do ii=1, ncell(1)
                           s = s + src(ii,jj,kk) / norm2([real(i-ii, R_P), real(j-jj, R_P), real(k-kk, R_P)] * dxyz)
                        enddo
                     enddo
                  enddo
                  gref = -FACTOR * product(dxyz) * s / (4._R_P * PI)
                  gmax = max(gmax, abs(uo(i,j,k) - gref)) ; gscale = max(gscale, abs(gref))
               enddo
            enddo
         enddo
      enddo
   enddo
   gmax = gmax / gscale
   ! D(G) of the solution: the free slabs as returned, the test's rules on the other faces
   bc_rule = merge(Z, bc, bc == F)
   ua = uo
   call rule_ghosts(ua, ncell, bc_rule, NO_NULL, gw)
   do ax=1, 3
      do side=1, 2
         if (bc(side,ax) /= F) cycle
         lo = 1 ; hi = ncell
         if (side == 1) then
            lo(ax) = 1 - gw ; hi(ax) = 0
         else
            lo(ax) = ncell(ax) + 1 ; hi(ax) = ncell(ax) + gw
         endif
         ua(lo(1):hi(1),lo(2):hi(2),lo(3):hi(3)) = uo(lo(1):hi(1),lo(2):hi(2),lo(3):hi(3))
      enddo
   enddo
   call d1d1(ua, ncell, dxyz, NO_NULL, p_, y)
   res = maxval(abs(y - FACTOR * src)) / maxval(abs(FACTOR * src))
   ok = gmax <= TOL_DISCRETE .and. res <= TOL_FREE .and. it(1) <= MAX_PCG_IT
   call leg_add(gmax, res, ok, it(1))
   if (c%rank == 0) print '(A)', '   D(G) p = '//itoa(p_)//' '//label//': free ghosts against the direct sum '// &
      fmt(gmax)//', |D(G) phi - factor f| '//fmt(res)//', CG iterations '//itoa(it(1))//merge(' ok  ', ' FAIL', ok)
   call solver%destroy
   endsubroutine check_free

   subroutine check_prism_ops(label, bc, s_)
   !< E3a: the manufactured `f = div(grad u)` by PRISM's centred FD and FV operators of order 2 s_ (GIDEON's; u's ghosts
   !< by the test's rules), solved (one 16^3 block, on rank 0), then `div(grad phi)` by the same operators from the
   !< solution and its returned ghosts: equal to f.
   character(*), intent(in) :: label   !< Configuration label.
   integer(I4P), intent(in) :: bc(2,3) !< Kinds per face [lo/hi, axis].
   integer(I4P), intent(in) :: s_      !< Half stencil (order 2 s_).
   class(elisha_spectral_solver_object), allocatable :: solver !< The solver.
   integer(I4P), parameter   :: NC=16_I4P   !< Cells per axis.
   integer(I4P), parameter   :: NCV(3)=[NC, NC, NC] !< Cells per axis, all three (a named actual argument).
   character(2), parameter   :: SCHEME(2)=['FD', 'FV'] !< Schemes.
   integer(I4P)              :: p_, sch, nbl, i, j, k, cc !< Order, scheme, counters.
   real(R_P)                 :: dxyz(3), res(2), fmax !< Spacing, residual per scheme, max |f|.
   real(R_P),    allocatable :: u(:,:,:), f(:,:,:), ph(:,:,:), buf(:), out(:) !< Fields.
   integer(I4P)              :: coords(3,1) !< The one block.
   logical                   :: ok          !< Pass flag.

   call new_fft(solver)
   p_ = 2 * s_
   dxyz = L / NC
   allocate(u(1-p_:NC+p_,1-p_:NC+p_,1-p_:NC+p_), f(NC,NC,NC), ph(1-p_:NC+p_,1-p_:NC+p_,1-p_:NC+p_))
   coords(:,1) = 0
   nbl = merge(1, 0, c%rank == 0)
   allocate(buf(max(nbl*NC**3,1)), out(max(nbl*(NC+2*p_)**3,1)))
   res = 0._R_P
   do sch=1, 2
      u = 0._R_P
      do k=1, NC
         do j=1, NC
            do i=1, NC
               u(i,j,k) = sin(0.7_R_P * i + 0.3_R_P) * cos(0.5_R_P * j - 0.2_R_P) * cos(0.4_R_P * k + 0.1_R_P) + &
                          0.25_R_P * cos(1.1_R_P * i - 0.9_R_P * j + 0.3_R_P * k)
            enddo
         enddo
      enddo
      call rule_ghosts(u, NCV, bc, NO_NULL, p_)
      call div_grad(u, s_, NC, dxyz, sch, f)
      cc = 0
      do k=1, NC
         do j=1, NC
            do i=1, NC
               cc = cc + 1
               buf(min(cc,size(buf))) = f(i,j,k) / FACTOR
            enddo
         enddo
      enddo
      call solver%initialize_blocks(nblk=B111, nijk=NCV, ngc=p_, dxyz=dxyz, level=0_I4P,                    &
                                    null_xyz=NO_NULL, isolated=.false., comm=c, stat=stat, &
                                    errmsg=msg, bc=bc, dg_order=p_)
      if (failed('initialize_blocks')) return
      out = 0._R_P
      call solver%solve_blocks(nbl=nbl, coords=coords, buf=buf, factor=FACTOR, gw=p_, out=out, stat=stat, errmsg=msg)
      if (failed('solve_blocks')) return
      call solver%destroy
      if (c%rank == 0) then
         cc = 0
         do k=1-p_, NC+p_
            do j=1-p_, NC+p_
               do i=1-p_, NC+p_
                  cc = cc + 1
                  ph(i,j,k) = out(cc)
               enddo
            enddo
         enddo
         call div_grad(ph, s_, NC, dxyz, sch, u(1:NC,1:NC,1:NC))
         fmax = maxval(abs(f))
         res(sch) = maxval(abs(u(1:NC,1:NC,1:NC) - f)) / fmax
      endif
   enddo
   call c%max_real8(res, stat) ! rank 0 holds them, the others 0
   ok = all(res <= merge(TOL_PCG, TOL_PRISM, any(bc == Z)))
   call leg_add(res(1), res(2), ok)
   if (c%rank == 0) print '(A)', '   order '//itoa(p_)//' '//label//': |div(grad phi) - f| / max|f| '//SCHEME(1)//' '// &
      fmt(res(1))//', '//SCHEME(2)//' '//fmt(res(2))//merge(' ok  ', ' FAIL', ok)
   endsubroutine check_prism_ops

   subroutine div_grad(q, s_, nc, dxyz, sch_, y)
   !< `y = div(grad q)` on the interior by GIDEON's centred operators (FD: sch_ = 1, FV: 2) of order 2 s_: the gradient
   !< on the interior and s_ ghost layers (from q's 2 s_ ghost layers), then the divergence (E3a).
   integer(I4P), intent(in)  :: s_, nc               !< Half stencil, cells per axis.
   real(R_P),    intent(in)  :: dxyz(3)              !< Spacing.
   real(R_P),    intent(in)  :: q(1-2*s_:,1-2*s_:,1-2*s_:) !< Padded scalar.
   integer(I4P), intent(in)  :: sch_                 !< Scheme.
   real(R_P),    intent(out) :: y(1:,1:,1:)          !< Result [nc^3].
   real(R_P), allocatable    :: gr(:,:,:,:)          !< Gradient, padded by s_.
   real(R_P)                 :: sx(-4:4), sy(-4:4), sz(-4:4) !< The stencils along x, y, z, contiguous: GIDEON's
                                                     !< explicit-shape dummies take no strided section (ifx's
                                                     !< -check arg_temp_created reports each copy).
   real(R_P)                 :: g3(3)                !< A gradient.
   integer(I4P)              :: ii, jj, kk, o        !< Counters.

   allocate(gr(3,1-s_:nc+s_,1-s_:nc+s_,1-s_:nc+s_))
   do kk=1-s_, nc+s_
      do jj=1-s_, nc+s_
         do ii=1-s_, nc+s_
            do o=-s_, s_
               sx(o) = q(ii+o,jj,kk) ; sy(o) = q(ii,jj+o,kk) ; sz(o) = q(ii,jj,kk+o)
            enddo
            if (sch_ == 1) then
               call gideon_fd_gradient_c(2*s_, sx(-s_:s_), sy(-s_:s_), sz(-s_:s_), dxyz, g3)
            else
               call gideon_fv_gradient_c(2*s_, sx(-s_:s_), sy(-s_:s_), sz(-s_:s_), dxyz, g3)
            endif
            gr(:,ii,jj,kk) = g3
         enddo
      enddo
   enddo
   do kk=1, nc
      do jj=1, nc
         do ii=1, nc
            do o=-s_, s_
               sx(o) = gr(1,ii+o,jj,kk) ; sy(o) = gr(2,ii,jj+o,kk) ; sz(o) = gr(3,ii,jj,kk+o)
            enddo
            if (sch_ == 1) then
               call gideon_fd_divergence_c(2*s_, sx(-s_:s_), sy(-s_:s_), sz(-s_:s_), dxyz, y(ii,jj,kk))
            else
               call gideon_fv_divergence_c(2*s_, sx(-s_:s_), sy(-s_:s_), sz(-s_:s_), dxyz, y(ii,jj,kk))
            endif
         enddo
      enddo
   enddo
   endsubroutine div_grad

   subroutine check_zero_compact
   !< E2b: the compact operator with zero-ghost faces is the one with ghost-Dirichlet faces (one ghost layer read): equal
   !< solutions, every cell and ghost, bitwise.
   class(elisha_spectral_solver_object), allocatable :: s1, s2 !< The solvers.
   real(R_P),    allocatable :: buf(:), o1(:), o2(:) !< Source, the two potentials.
   integer(I4P), allocatable :: coords(:,:)          !< This rank's blocks.
   integer(I4P)              :: nbl, i, ndiff(1)     !< Blocks, counter, differing values.
   real(R_P)                 :: dmax(1)              !< Max difference.
   logical                   :: ok                   !< Pass flag.

   call new_fft(s1)
   call new_fft(s2)
   call round_robin(B222, coords, nbl)
   allocate(buf(max(nbl*512,1)), o1(max(nbl*1000,1)), o2(max(nbl*1000,1)))
   do i=1, size(buf)
      buf(i) = sin(0.37_R_P * i + c%rank)
   enddo
   o1 = 0._R_P ; o2 = 0._R_P
   bcx = reshape([Z,Z, D,N, P,P], [2,3])
   call s1%initialize_blocks(nblk=B222, nijk=C888, ngc=1_I4P, dxyz=DX01, level=0_I4P, &
                             null_xyz=NO_NULL, isolated=.false., comm=c, stat=stat, errmsg=msg, &
                             bc=bcx)
   if (failed('initialize_blocks (Z)')) return
   bcx = reshape([G,G, D,N, P,P], [2,3])
   call s2%initialize_blocks(nblk=B222, nijk=C888, ngc=1_I4P, dxyz=DX01, level=0_I4P, &
                             null_xyz=NO_NULL, isolated=.false., comm=c, stat=stat, errmsg=msg, &
                             bc=bcx)
   if (failed('initialize_blocks (G)')) return
   call s1%solve_blocks(nbl=nbl, coords=coords, buf=buf, factor=FACTOR, gw=1_I4P, out=o1, stat=stat, errmsg=msg)
   if (failed('solve_blocks (Z)')) return
   call s2%solve_blocks(nbl=nbl, coords=coords, buf=buf, factor=FACTOR, gw=1_I4P, out=o2, stat=stat, errmsg=msg)
   if (failed('solve_blocks (G)')) return
   dmax = 0._R_P ; ndiff = 0
   if (nbl > 0) then
      dmax = maxval(abs(o1(1:nbl*1000) - o2(1:nbl*1000)))
      ndiff = count(transfer(o1(1:nbl*1000), 0_I8P, nbl*1000) /= transfer(o2(1:nbl*1000), 0_I8P, nbl*1000), kind=I4P)
   endif
   call c%max_real8(dmax, stat)
   call c%max_i4(ndiff, stat)
   ok = ndiff(1) == 0
   call leg_add(dmax(1), 0._R_P, ok)
   if (c%rank == 0) print '(A)', '   compact operator, Z/Z D/N P/P against G/G D/N P/P (every cell and ghost): '// &
      fmt(dmax(1))//', values differing '//itoa(ndiff(1))//merge(' ok  ', ' FAIL', ok)
   call s1%destroy
   call s2%destroy
   endsubroutine check_zero_compact

   subroutine check_refusal(name)
   !< A configuration ELISHA must refuse: `initialize_blocks` returns the expected status, the same on every rank.
   character(*), intent(in) :: name !< The refusal (ADAM's flag without --refuse-).
   class(elisha_spectral_solver_object), allocatable :: solver !< The solver.
   integer(I4P)              :: expected, st(1), lo(1), hi(1) !< Expected status, status and its range over the ranks.
   logical                   :: ok      !< Pass flag.
   character(:), allocatable :: label, m_ !< Label, message.

   call new_fft(solver)
   select case(name)
   case('mixed-periodic')
      label = 'a periodic face against a Dirichlet one' ; expected = ELISHA_ERR_BC_PERIODIC_PAIR
      bcx = reshape([P,D, D,D, D,D], [2,3])
      call solver%initialize_blocks(nblk=B222, nijk=C888, ngc=1_I4P, dxyz=DXL16, level=0_I4P, &
                                    null_xyz=NO_NULL, isolated=.false., comm=c, stat=st(1), &
                                    errmsg=m_, bc=bcx)
   case('mixed-ghost')
      label = 'a ghost-Dirichlet face against a face kind' ; expected = ELISHA_ERR_BC_GHOST_PAIR
      bcx = reshape([G,D, D,D, D,D], [2,3])
      call solver%initialize_blocks(nblk=B222, nijk=C888, ngc=1_I4P, dxyz=DXL16, level=0_I4P, &
                                    null_xyz=NO_NULL, isolated=.false., comm=c, stat=st(1), &
                                    errmsg=m_, bc=bcx)
   case('wide-isolated')
      label = 'the wide D(G) with isolated boundaries' ; expected = ELISHA_ERR_ISOLATED_WIDE
      call solver%initialize_blocks(nblk=B222, nijk=C888, ngc=4_I4P, dxyz=DX01, &
                                    level=0_I4P, null_xyz=NO_NULL, isolated=.true., comm=c, &
                                    stat=st(1), errmsg=m_, dg_order=4_I4P)
   case('zero-mixed')
      label = 'a zero-ghost face against a Dirichlet one' ; expected = ELISHA_ERR_BC_ZERO_GHOST_PAIR
      bcx = reshape([Z,D, P,P, P,P], [2,3])
      call solver%initialize_blocks(nblk=B222, nijk=C888, ngc=4_I4P, dxyz=DXL16, level=0_I4P, &
                                    null_xyz=NO_NULL, isolated=.false., comm=c, stat=st(1), &
                                    errmsg=m_, bc=bcx, dg_order=4_I4P)
   case('free-compact')
      label = 'free-space ghosts with the compact operator' ; expected = ELISHA_ERR_BC_FREE_GHOSTS
      bcx = reshape([F,F, F,F, F,F], [2,3])
      call solver%initialize_blocks(nblk=B222, nijk=C444, ngc=1_I4P, dxyz=DXL8, level=0_I4P, &
                                    null_xyz=NO_NULL, isolated=.false., comm=c, stat=st(1), &
                                    errmsg=m_, bc=bcx, dg_order=0_I4P)
   endselect
   lo = st ; hi = st
   call c%min_i4(lo, stat) ; call c%max_i4(hi, stat)
   ok = lo(1) == expected .and. hi(1) == expected
   if (c%rank == 0) then
      print '(A)', '   refusal, '//label//': '//elisha_status_name(st(1))//' (expected '//elisha_status_name(expected)// &
                   ')  '//merge('PASS', 'FAIL', ok)
      if (allocated(m_)) print '(A)', '      '//m_
   endif
   test_passed = test_passed .and. ok
   if (nrows < MAX_ROWS) then
      nrows = nrows + 1
      rows(nrows) = 'refusal, '//label//': '//elisha_status_name(st(1))//' | '//merge('PASS', 'FAIL', ok)
   endif
   call solver%destroy
   endsubroutine check_refusal

   ! ---------------------------------------------------------------------------------------------------------------------
   ! the eigenfunctions, the rules, the operators (ADAM's, D1 from GIDEON's rows)
   ! ---------------------------------------------------------------------------------------------------------------------
   subroutine check_zero_not_converged
   !< E2b (ELISHA notes E1c, decision C): the CG of D(G) p = 4, Z/Z D/D N/N, stopped at max_iterations = 1 above its
   !< tolerance: ELISHA_ERR_NOT_CONVERGED on every rank, `converged` false, `out` the last iterate (finite, ghosts
   !< included); then max_iterations = 200: converged.
   integer(I4P), parameter :: P4=4_I4P, NB(3)=[4_I4P, 4_I4P, 4_I4P], NC(3)=[4_I4P, 4_I4P, 4_I4P] !< Order, blocks, cells.
   class(elisha_spectral_solver_object), allocatable :: solver !< The solver.
   integer(I4P), allocatable :: coords(:,:) !< This rank's block coordinates.
   real(R_P),    allocatable :: buf(:), out(:) !< Packed source and potential.
   integer(I4P)              :: nbl, i, bad(1) !< Blocks of this rank, counter, non-finite values over the ranks.
   logical                   :: ok           !< Pass flag.

   call new_fft(solver)
   call round_robin(NB, coords, nbl)
   allocate(buf(max(nbl*product(NC),1)), out(max(nbl*product(NC+2*P4),1)))
   do i=1, size(buf)
      buf(i) = sin(0.37_R_P * i + 1.1_R_P * c%rank) ! no smooth structure: one CG step cannot solve it
   enddo
   call solver%initialize_blocks(nblk=NB, nijk=NC, ngc=P4, dxyz=L/(NB*NC), level=0_I4P, null_xyz=[.false., .false., &
                                 .false.], isolated=.false., comm=c, stat=stat, errmsg=msg,                        &
                                 bc=reshape([Z,Z, D,D, N,N], [2,3]), dg_order=P4)
   if (failed('initialize_blocks')) return
   solver%max_iterations = 1_I4P
   out = 0._R_P
   call solver%solve_blocks(nbl=nbl, coords=coords, buf=buf, factor=FACTOR, gw=P4, out=out, stat=stat, errmsg=msg)
   ok = stat == ELISHA_ERR_NOT_CONVERGED .and. .not.solver%converged
   bad = 0
   if (any(out /= out) .or. any(abs(out) > huge(1._R_P))) bad = 1
   call c%max_i4(bad, i)
   ok = ok .and. bad(1) == 0
   if (c%rank == 0) print '(A)', '   p = 4, Z/Z D/D N/N, max_iterations 1: '//elisha_status_name(stat)// &
      ', the last iterate '//trim(merge('finite    ', 'NOT FINITE', bad(1) == 0))//merge(' ok  ', ' FAIL', ok)
   solver%max_iterations = 200_I4P
   call solver%solve_blocks(nbl=nbl, coords=coords, buf=buf, factor=FACTOR, gw=P4, out=out, stat=stat, errmsg=msg)
   ok = ok .and. stat == ELISHA_OK .and. solver%converged
   if (c%rank == 0) print '(A)', '   p = 4, Z/Z D/D N/N, max_iterations 200: '//elisha_status_name(stat)// &
      merge(' ok  ', ' FAIL', stat == ELISHA_OK .and. solver%converged)
   lok = lok .and. ok
   call solver%destroy
   endsubroutine check_zero_not_converged

   subroutine round_robin(nblk, coords, nbl)
   !< This rank's blocks of an nblk lattice (x fastest), dealt round robin [3,max(nbl,1)].
   integer(I4P),              intent(in)    :: nblk(3)     !< Blocks per direction.
   integer(I4P), allocatable, intent(inout) :: coords(:,:) !< This rank's block coordinates.
   integer(I4P),              intent(out)   :: nbl         !< This rank's blocks.
   integer(I4P)                             :: b           !< Counter.

   nbl = 0
   do b=0, product(nblk)-1
      if (mod(b, c%nranks) == c%rank) nbl = nbl + 1
   enddo
   if (allocated(coords)) deallocate(coords)
   allocate(coords(3,max(nbl,1)))
   coords = 0
   nbl = 0
   do b=0, product(nblk)-1
      if (mod(b, c%nranks) /= c%rank) cycle
      nbl = nbl + 1
      coords(:,nbl) = [mod(b, nblk(1)), mod(b / nblk(1), nblk(2)), b / (nblk(1) * nblk(2))]
   enddo
   endsubroutine round_robin

   function mode(gi, gj, gk) result(v)
   !< The product of the axis eigenfunctions at the global cell (gi, gj, gk), ghosts included.
   integer(I4P), intent(in) :: gi, gj, gk !< Global cell indices.
   real(R_P)                :: v          !< Value.

   v = axis_fun(cfg_bc(:,1), cfg_kmode(1), cfg_ncell(1), gi, cfg_null(1)) * &
       axis_fun(cfg_bc(:,2), cfg_kmode(2), cfg_ncell(2), gj, cfg_null(2)) * &
       axis_fun(cfg_bc(:,3), cfg_kmode(3), cfg_ncell(3), gk, cfg_null(3))
   endfunction mode

   function source(gi, gj, gk) result(v)
   !< The source at an interior cell: the mode, plus RHO0 without a Dirichlet axis (the solve must drop it).
   integer(I4P), intent(in) :: gi, gj, gk !< Global cell indices.
   real(R_P)                :: v          !< Value.

   v = mode(gi, gj, gk)
   if (.not.cfg_dirichlet) v = v + RHO0
   endfunction source

   subroutine axis_eigen(kinds, k, n_, dx, is_null, dg_order, mu, kap)
   !< The discrete eigenvalue and the continuum wavenumber of an axis's eigenfunction: the 7-point operator's in closed
   !< form, or the wide D(G)'s in real space (D1 applied twice to the eigenfunction, extended by its own formula over
   !< the ghost cells, divided by it at the cell where it is largest), independently of the solver's symbol.
   integer(I4P), intent(in)  :: kinds(2) !< Kinds at the two faces.
   integer(I4P), intent(in)  :: k        !< Mode.
   integer(I4P), intent(in)  :: n_       !< Cells.
   real(R_P),    intent(in)  :: dx       !< Spacing.
   logical,      intent(in)  :: is_null  !< Null direction.
   integer(I4P), intent(in)  :: dg_order !< 0 compact, p the wide D(G).
   real(R_P),    intent(out) :: mu, kap  !< Discrete eigenvalue, continuum wavenumber.
   real(R_P)                 :: th       !< Phase step per cell.
   real(R_P)                 :: gg(-4:4) !< D1 of the eigenfunction around the cell.
   real(R_P)                 :: dg, fmax !< D1 D1 at the cell, the eigenfunction there.
   integer(I4P)              :: i0, i, j, o !< Cell, counters.

   mu = 0._R_P ; kap = 0._R_P
   if (is_null) return
   th = phase_step(kinds, k, n_)
   kap = th / dx
   if (dg_order == 0_I4P) then
      mu = (2._R_P * cos(th) - 2._R_P) / dx**2
      return
   endif
   i0 = 1 ; fmax = 0._R_P
   do i=1, n_
      if (abs(axis_fun(kinds, k, n_, i, .false.)) > fmax) then
         fmax = abs(axis_fun(kinds, k, n_, i, .false.)) ; i0 = i
      endif
   enddo
   gg = 0._R_P
   do o=-dg_order/2, dg_order/2
      do j=1, dg_order/2
         gg(o) = gg(o) + a_d1(j,dg_order/2) * (axis_fun(kinds, k, n_, i0+o+j, .false.) - &
                                               axis_fun(kinds, k, n_, i0+o-j, .false.))
      enddo
   enddo
   dg = 0._R_P
   do j=1, dg_order/2
      dg = dg + a_d1(j,dg_order/2) * (gg(j) - gg(-j))
   enddo
   mu = dg / axis_fun(kinds, k, n_, i0, .false.) / dx**2
   endsubroutine axis_eigen

   pure function phase_step(kinds, k, n_) result(th)
   !< The eigenfunction's phase step per cell of an axis.
   integer(I4P), intent(in) :: kinds(2) !< Kinds at the two faces.
   integer(I4P), intent(in) :: k, n_    !< Mode, cells.
   real(R_P)                :: th       !< Phase step.

   if (all(kinds == P)) then
      th = 2._R_P * PI * k / n_
   elseif (all(kinds == G)) then
      th = PI * k / (n_ + 1)
   elseif (kinds(1) == kinds(2)) then
      th = PI * k / n_
   else
      th = PI * (k + 0.5_R_P) / n_
   endif
   endfunction phase_step

   pure function axis_fun(kinds, k, n_, gi, is_null) result(v)
   !< The axis eigenfunction at the global cell index gi (1..n interior, beyond: the ghosts).
   integer(I4P), intent(in) :: kinds(2)  !< Kinds at the two faces.
   integer(I4P), intent(in) :: k, n_, gi !< Mode, cells, index.
   logical,      intent(in) :: is_null   !< Null direction.
   real(R_P)                :: v         !< Value.
   real(R_P)                :: th, x     !< Phase step, cell phase.

   v = 1._R_P
   if (is_null) return
   th = phase_step(kinds, k, n_)
   if (all(kinds == G)) then
      v = sin(th * gi)
      return
   endif
   x = th * (gi - 0.5_R_P)
   if (all(kinds == P)) then
      v = cos(x)
   elseif (kinds(1) == D) then
      v = sin(x)
   else
      v = cos(x)
   endif
   endfunction axis_fun

   subroutine rule_ghosts(u, nc, bc, null_xyz, ng)
   !< The `ng` ghost layers of the padded global array `u` by the faces' rules (the test's own): periodic (and null) the
   !< wrap, Dirichlet and Neumann at the face the odd and even reflections about it, ghost-Dirichlet the odd reflection
   !< about the ghost centre, zero ghosts 0.
   integer(I4P), intent(in)    :: nc(3), bc(2,3), ng   !< Cells, kinds, ghost layers.
   logical,      intent(in)    :: null_xyz(3)          !< Null directions.
   real(R_P),    intent(inout) :: u(1-ng:,1-ng:,1-ng:) !< Padded array.
   integer(I4P)                :: ax, j, lo, hi        !< Counters.

   do ax=1, 3
      do j=1, ng
         lo = 1 - j ; hi = nc(ax) + j
         if (null_xyz(ax) .or. all(bc(:,ax) == P)) then
            call put(u, ng, ax, lo, modulo(lo - 1, nc(ax)) + 1, 1._R_P)
            call put(u, ng, ax, hi, modulo(hi - 1, nc(ax)) + 1, 1._R_P)
            cycle
         endif
         select case(bc(1,ax))
         case(D) ; call put(u, ng, ax, lo, j, -1._R_P)
         case(N) ; call put(u, ng, ax, lo, j, 1._R_P)
         case(G)
            if (j == 1) then
               call put(u, ng, ax, lo, 1, 0._R_P)
            else
               call put(u, ng, ax, lo, j-1, -1._R_P)
            endif
         case default ; call put(u, ng, ax, lo, 1, 0._R_P)
         endselect
         select case(bc(2,ax))
         case(D) ; call put(u, ng, ax, hi, nc(ax)+1-j, -1._R_P)
         case(N) ; call put(u, ng, ax, hi, nc(ax)+1-j, 1._R_P)
         case(G)
            if (j == 1) then
               call put(u, ng, ax, hi, nc(ax), 0._R_P)
            else
               call put(u, ng, ax, hi, nc(ax)+2-j, -1._R_P)
            endif
         case default ; call put(u, ng, ax, hi, nc(ax), 0._R_P)
         endselect
      enddo
   enddo
   endsubroutine rule_ghosts

   subroutine put(u, ng, d_, t, src, sg)
   !< Plane t of axis d_ of the padded array u <- sg times plane src.
   integer(I4P), intent(in)    :: ng         !< Ghost layers.
   real(R_P),    intent(inout) :: u(1-ng:,1-ng:,1-ng:) !< Padded array.
   integer(I4P), intent(in)    :: d_, t, src !< Axis, indices.
   real(R_P),    intent(in)    :: sg         !< Sign.

   select case(d_)
   case(1) ; u(t,:,:) = sg * u(src,:,:)
   case(2) ; u(:,t,:) = sg * u(:,src,:)
   case(3) ; u(:,:,t) = sg * u(:,:,src)
   endselect
   endsubroutine put

   subroutine d1d1(u, nc, dxyz, null_xyz, p_, y)
   !< `y = D(G) u` on the interior: per axis D1 (GIDEON's coefficients) applied to the padded u (its ghosts set), then
   !< D1 again.
   integer(I4P), intent(in)  :: nc(3), p_            !< Cells, order.
   real(R_P),    intent(in)  :: u(1-p_:,1-p_:,1-p_:) !< Padded array.
   real(R_P),    intent(in)  :: dxyz(3)              !< Spacing.
   logical,      intent(in)  :: null_xyz(3)          !< Null directions.
   real(R_P),    intent(out) :: y(1:,1:,1:)          !< Result [nc].
   real(R_P), allocatable    :: gg(:)                !< D1 along a line.
   integer(I4P)              :: ax, i, j, k, o, a_, idx(3) !< Counters.

   y = 0._R_P
   allocate(gg(-p_/2:p_/2))
   do ax=1, 3
      if (null_xyz(ax)) cycle
      do k=1, nc(3)
         do j=1, nc(2)
            do i=1, nc(1)
               do o=-p_/2, p_/2
                  gg(o) = 0._R_P
                  do a_=1, p_/2
                     idx = [i, j, k] ; idx(ax) = idx(ax) + o + a_
                     gg(o) = gg(o) + a_d1(a_,p_/2) * u(idx(1),idx(2),idx(3))
                     idx(ax) = idx(ax) - 2 * a_
                     gg(o) = gg(o) - a_d1(a_,p_/2) * u(idx(1),idx(2),idx(3))
                  enddo
               enddo
               do a_=1, p_/2
                  y(i,j,k) = y(i,j,k) + a_d1(a_,p_/2) * (gg(a_) - gg(-a_)) / dxyz(ax)**2
               enddo
            enddo
         enddo
      enddo
   enddo
   endsubroutine d1d1

   function op_label() result(lab)
   !< The operator of the current case, for the lines.
   character(:), allocatable :: lab !< Label.

   if (cfg_order == 0) then
      lab = 'compact   '
   else
      lab = 'D(G) p = '//itoa(cfg_order)
   endif
   endfunction op_label

   logical function failed(what)
   !< A step returned an error: report it, fail the test.
   character(*), intent(in) :: what !< The step.

   failed = stat /= ELISHA_OK
   if (.not.failed) return
   test_passed = .false. ; lok = .false.
   if (c%rank == 0) then
      if (allocated(msg)) then
         print '(A)', '   '//what//' failed: '//msg
      else
         print '(A)', '   '//what//' failed: '//elisha_status_name(stat)
      endif
   endif
   endfunction failed

   pure function itoa(i) result(s)
   !< An integer as a trimmed string.
   integer(I4P), intent(in)  :: i !< Value.
   character(:), allocatable :: s !< String.
   character(16)             :: w !< Buffer.

   write(w, '(I0)') i
   s = trim(w)
   endfunction itoa

   pure function fmt(x) result(s)
   !< A real in scientific notation, trimmed.
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
endprogram elisha_sv_bc_mpi
