!< ELISHA verification SV-2 and SV-6: the spectral solver against discrete and continuum solutions (ADAM's test_poisson_fft).
program elisha_sv_fft_mpi
!< ELISHA verification SV-2 and SV-6: the spectral solver against discrete and continuum solutions (ADAM's test_poisson_fft).
!<
!< **Provenance.** ADAM's `src/tests/poisson/test_poisson_fft.F90` (ADAM issue #81, P2 SV-2 and P4 SV-6; the slab leg of
!< issue #84, E5d), ported onto ELISHA: the cases, the sources, the exact solutions, the measures and the bounds are
!< ADAM's; the solver is `elisha_spectral_solver_object` (ADAM's `poisson_fft_object`), the grid an
!< `elisha_mesh_object` built here (ADAM read an input file through its realm), and a refusal is the status code the
!< setup returns (`elisha_status_name`), not a stop with FLAIL's message.
!<
!< **Why.** Self-gravity conserves E + rho Phi and momentum exactly only if Phi solves the discrete Poisson equation
!< `L_h Phi = 4 pi G (rho - mean rho)` to round-off. The solver divides each Fourier mode by the eigenvalue of `L_h`; a
!< wrong eigenvalue (a missing factor, a swapped axis, a wrong spacing), a misplaced block in the gather, or a wrong
!< ghost wrap in the scatter each break that.
!<
!< **SV-2, periodic.** The source is `rho = rho0 + A1 sin(k1 x) cos(k1 y) cos(k1 z) + A2 cos(k2 . x)`, two oblique modes
!< on the anisotropic periodic box [0,1] x [0,2] x [0,3]. Each mode is an eigenfunction of `L_h`, so the discrete
!< solution is known in closed form (each mode divided by its discrete eigenvalue, the mean dropped), and so is the
!< continuum one (divided by `-|k|^2`). Asserted, per configuration and level:
!<   - discrete: `max |Phi - Phi_h| / max |Phi_h|` over every cell, ghost layers included, at round-off (`TOL_DISCRETE`);
!<   - residual: `max |L_h Phi - 4 pi G (rho - rho0)| / max |4 pi G (rho - rho0)|` over the interior, from the returned
!<     ghosts (`TOL_RESIDUAL`);
!<   - continuum: the max error against the exact Phi, second order on the finest pair (`MIN_ORDER`);
!<   - no ghost cell left unfilled (a sentinel).
!< Configurations: an octree in 3-D (levels 1-3, 16^3 to 64^3 cells), a quadtree in 2-D (null z, levels 1-3) and a
!< single block (level 0: the wrap inside one block; on more than one rank, ranks without blocks). 8 cells per block
!< side, 3 ghost layers.
!<
!< **SV-6, isolated.** On the cube [-1,1]^3 (octree, 16^3 to 64^3 cells), two spheres of radius `A_SPHERE` centred at
!< the origin, phi -> 0 at infinity:
!<   - the n = 1 polytrope, `rho = sin(k r) / (k r)`, `k = pi / a`, exact `Phi = -(4 pi G / k^2) sin(k r) / (k r) - G M / a`
!<     inside, `-G M / r` outside, `M = 4 a^3 / pi`;
!<   - the uniform sphere, `rho = 1` (cell averages, `SUB^3` samples per cell), exact `-2 pi G (a^2 - r^2 / 3)` inside,
!<     `-G M / r` outside, `M = 4 pi a^3 / 3`.
!< Asserted: the residual `max |L_h Phi - 4 pi G rho| / max |4 pi G rho|` over the interior (`TOL_RES_ISOLATED`: the
!< lattice kernel's own residual summed over the source cells); the max error against the exact Phi over every cell,
!< ghosts included, and its least-squares order over the ladder (`MIN_ORDER` for the polytrope, `MIN_ORDER_SPHERE` for
!< the uniform sphere, whose surface cuts cells); no ghost left unfilled.
!<
!< **Refusals** (each asserted as a status, the same on every rank): SV-2, a grid without periodic boundaries
!< (`ELISHA_ERR_BC_MISSING`); SV-6, a periodic grid (`ELISHA_ERR_ISOLATED_PERIODIC`), cells 1/16 x 2/16 x 3/16
!< (`ELISHA_ERR_ISOLATED_CELLS`), a 2-D grid (`ELISHA_ERR_ISOLATED_DIMENSION`).
!<
!< **Rank invariance** (ADAM's slab leg). On more than one rank, after every solve rank 0 solves the same level again
!< alone (on `MPI_COMM_SELF`, every block its own) and compares the transform array of both solves
!< (`global_potential`: images and padding included) bit for bit: the distributed slab transform must not depend on
!< the rank count. `--dump DIR` also writes rank 0's transform array of every level to
!< `DIR/phi-r<ratio>-nk<nk>-l<level>-p<profile>.bin` (ADAM's names and format: 3 int32 sizes, then the float64 values,
!< x fastest; ratio 8 octree, 4 quadtree; profile 0 periodic, 1 polytrope, 2 uniform sphere), for comparing the dumps
!< of different rank counts or builds byte for byte (`poisson_oracle.py compare`).
!<
!< Output: per level ADAM's line (`<label> N = <n>: |Phi - Phi_h| <e>, residual <e>, |Phi - Phi_exact| <e>, unfilled
!< <n>`, which ADAM's figure scripts parse), per ladder the order line and the verdict, then a summary table with the
!< bounds and `elisha_sv_fft_mpi: PASSED` or `FAILED` (exit 1).
!<
!< Usage: `mpirun -np N exe/elisha_sv_fft_mpi [--periodic | --isolated] [--dump DIR] [--no-rank-check]`
!<        `mpirun -np N exe/elisha_sv_fft_mpi --no-periodic | --refuse-isolated-{periodic,anisotropic,2d}`
!<   (no flag)          SV-2 and SV-6 with their refusals (the CI run);
!<   --periodic         SV-2 only (the three ladders and the non-periodic refusal);
!<   --isolated         SV-6 only (the two spheres and the three isolated refusals);
!<   --dump DIR         write rank 0's transform array of every level into DIR (it must exist);
!<   --no-rank-check    skip the rank-0 re-solve and its bitwise comparison;
!<   --no-periodic, --refuse-isolated-X   only that refusal: exit 0 when the setup returns the expected code (printed
!<                      by name), 1 otherwise. ADAM's run had to stop (exit 1); ELISHA returns a status instead.
!< `--device` (only in the device build, `tests-nvf-oac`, `exe-oac/`): the cuFFT backend
!< (`elisha_spectral_solver_cufft_object`), each rank on the GPU of its node-local rank; the same bounds, and the dumps
!< compare with the host build's to round-off (`scripts/verification/poisson_oracle.py compare`).
!< Built with `-DELISHA_NO_MPI` the program runs on one rank, without the rank-invariance check. The FP traps of a debug
!< build are off around `MPI_Init` only (hwloc raises IEEE flags inside it).
#ifndef ELISHA_NO_MPI
use mpi
#endif
#ifdef _CUFFT
! tests-nvf-oac: the device backends, selected at run time by --device
use openacc
use elisha_device, only : elisha_spectral_solver_cufft_object, elisha_hierarchical_solver_oac_object
#endif
use, intrinsic :: ieee_exceptions, only : ieee_usual, ieee_get_halting_mode, ieee_set_halting_mode
use elisha, only : I4P, I8P, R_P, ELISHA_OK, ELISHA_ERR_BC_MISSING, ELISHA_ERR_ISOLATED_PERIODIC,           &
                   ELISHA_ERR_ISOLATED_CELLS, ELISHA_ERR_ISOLATED_DIMENSION, elisha_status_name, elisha_comm_object, &
                   elisha_mesh_object, elisha_spectral_solver_object
implicit none
real(R_P),    parameter :: PI=acos(-1._R_P)              !< Pi.
real(R_P),    parameter :: FACTOR=4._R_P * PI            !< Source factor, 4 pi G with G = 1.
real(R_P),    parameter :: RHO0=1.5_R_P                  !< Mean density (dropped by the solve).
real(R_P),    parameter :: A1=1._R_P                     !< Amplitude of the first mode.
real(R_P),    parameter :: A2=0.5_R_P                    !< Amplitude of the second mode.
real(R_P),    parameter :: L(3)=[1._R_P, 2._R_P, 3._R_P] !< Domain lengths (SV-2).
real(R_P),    parameter :: SENTINEL=-1.e30_R_P           !< Value of a phi cell the solve did not fill.
real(R_P),    parameter :: TOL_DISCRETE=1.e-12_R_P       !< Bound on the error against the discrete solution, relative.
real(R_P),    parameter :: TOL_RESIDUAL=1.e-10_R_P       !< Bound on the discrete residual, relative.
real(R_P),    parameter :: MIN_ORDER=1.9_R_P             !< Bound on the order against the continuum solution.
real(R_P),    parameter :: A_SPHERE=0.5_R_P              !< Radius of the isolated spheres.
integer(I4P), parameter :: SUB=8_I4P                     !< Samples per direction of a cell average (uniform sphere).
real(R_P),    parameter :: TOL_RES_ISOLATED=1.e-10_R_P   !< Bound on the isolated residual, relative.
real(R_P),    parameter :: MIN_ORDER_SPHERE=1.2_R_P      !< Bound on the uniform sphere's order (least squares).
integer(I4P), parameter :: PROFILE_PERIODIC=0_I4P        !< The periodic modes.
integer(I4P), parameter :: PROFILE_POLYTROPE=1_I4P       !< The isolated n = 1 polytrope.
integer(I4P), parameter :: PROFILE_UNIFORM=2_I4P         !< The isolated uniform sphere.
integer(I4P), parameter :: NB_SIDE=8_I4P                 !< Cells per block side (ADAM's ni = nj = 8).
integer(I4P), parameter :: NGC=3_I4P                     !< Ghost layers (ADAM's ngc).
integer(I4P), parameter :: MAX_ROWS=32_I4P               !< Rows of the summary table.
type(elisha_comm_object)  :: c                           !< Communication context (the world).
logical                   :: test_passed                 !< Aggregate pass flag.
logical                   :: run_periodic, run_isolated  !< Suites to run.
logical                   :: rank_check                  !< Re-solve on rank 0 alone and compare bitwise.
character(64)             :: refusal                     !< The only refusal to run ('': none).
character(256)            :: arg                         !< Command argument.
character(256)            :: dump_dir                    !< Directory of the potential dumps ('': no dump).
character(200)            :: rows(MAX_ROWS)              !< Summary rows.
integer(I4P)              :: nrows                       !< Summary rows used.
integer(I4P)              :: a, stat                     !< Counter, status.
integer                   :: ierr, world                 !< MPI error status, the world communicator.
logical                   :: halting(3)                  !< The FP halting modes of the build (debug: on).
logical                   :: device                      !< --device: the device backends.
real(R_P)                 :: t0                          !< Wall clock at the start.
character(:), allocatable :: msg                         !< Message.

run_periodic = .true. ; run_isolated = .true. ; rank_check = .true.
refusal = '' ; dump_dir = '' ; nrows = 0
device = .false.
a = 0
do while (a < command_argument_count())
   a = a + 1
   call get_command_argument(a, arg)
   select case(trim(arg))
   case('--periodic')
      run_isolated = .false.
   case('--isolated')
      run_periodic = .false.
   case('--no-periodic')
      refusal = 'no-periodic'
   case('--no-rank-check')
      rank_check = .false.
   case('--dump')
      a = a + 1
      call get_command_argument(a, dump_dir)
   case('--device')   ; device = .true.
   case default
      if (index(arg, '--refuse-isolated-') == 1) then
         refusal = arg(19:)
      else
         print '(A)', 'elisha_sv_fft_mpi: unknown argument "'//trim(arg)//'"'
         error stop 2
      endif
   endselect
enddo
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
   print '(A)', 'elisha_sv_fft_mpi: --device needs the device build (tests-nvf-oac)'
   error stop 2
endif
#endif
call c%init(world, stat, msg)
if (stat /= ELISHA_OK) then
   print '(A)', 'elisha_sv_fft_mpi: comm init: '//msg
   error stop 1
endif
t0 = c%wtime()
test_passed = .true.
if (len_trim(refusal) > 0) then
   select case(trim(refusal))
   case('no-periodic')
      call check_refusal(label='3-D octree, not periodic', nk=8_I4P, level=1_I4P, periodic=.false., isolated=.false., &
                         cube=.false., expected=ELISHA_ERR_BC_MISSING)
   case('periodic')
      call check_refusal(label='isolated, periodic grid', nk=8_I4P, level=1_I4P, periodic=.true., isolated=.true., &
                         cube=.true., expected=ELISHA_ERR_ISOLATED_PERIODIC)
   case('anisotropic')
      call check_refusal(label='isolated, anisotropic cells', nk=8_I4P, level=1_I4P, periodic=.false., &
                         isolated=.true., cube=.false., expected=ELISHA_ERR_ISOLATED_CELLS)
   case('2d')
      call check_refusal(label='isolated, 2-D', nk=1_I4P, level=1_I4P, periodic=.false., isolated=.true., cube=.true., &
                         expected=ELISHA_ERR_ISOLATED_DIMENSION)
   case default
      if (c%rank == 0) print '(A)', 'elisha_sv_fft_mpi: unknown refusal "'//trim(refusal)//'"'
      call finish
      error stop 2
   endselect
else
   if (run_periodic) then
      if (c%rank == 0) print '(A)', 'SV-2: the spectral solver, periodic'
      call check_ladder(label='3-D octree,    ', nk=8_I4P, levels=[1_I4P, 2_I4P, 3_I4P], periodic=.true., &
                        prof=PROFILE_PERIODIC)
      call check_ladder(label='2-D quadtree,  ', nk=1_I4P, levels=[1_I4P, 2_I4P, 3_I4P], periodic=.true., &
                        prof=PROFILE_PERIODIC)
      call check_ladder(label='3-D one block, ', nk=8_I4P, levels=[0_I4P], periodic=.true., prof=PROFILE_PERIODIC)
      call check_refusal(label='3-D octree, not periodic', nk=8_I4P, level=1_I4P, periodic=.false., isolated=.false., &
                         cube=.false., expected=ELISHA_ERR_BC_MISSING)
   endif
   if (run_isolated) then
      if (c%rank == 0) print '(A)', 'SV-6: the spectral solver, isolated'
      call check_ladder(label='polytrope n=1, ', nk=8_I4P, levels=[1_I4P, 2_I4P, 3_I4P], periodic=.false., &
                        prof=PROFILE_POLYTROPE)
      call check_ladder(label='uniform sphere,', nk=8_I4P, levels=[1_I4P, 2_I4P, 3_I4P], periodic=.false., &
                        prof=PROFILE_UNIFORM)
      call check_refusal(label='isolated, periodic grid', nk=8_I4P, level=1_I4P, periodic=.true., isolated=.true., &
                         cube=.true., expected=ELISHA_ERR_ISOLATED_PERIODIC)
      call check_refusal(label='isolated, anisotropic cells', nk=8_I4P, level=1_I4P, periodic=.false., &
                         isolated=.true., cube=.false., expected=ELISHA_ERR_ISOLATED_CELLS)
      call check_refusal(label='isolated, 2-D', nk=1_I4P, level=1_I4P, periodic=.false., isolated=.true., cube=.true., &
                         expected=ELISHA_ERR_ISOLATED_DIMENSION)
   endif
endif
if (c%rank == 0) then
   print '(A,I0,A,F8.2,A)', 'elisha_sv_fft_mpi summary (np ', c%nranks, ', wall ', c%wtime() - t0, ' s):'
   do a=1, nrows
      print '(A)', '  '//trim(rows(a))
   enddo
   if (test_passed) then
      print '(A)', 'elisha_sv_fft_mpi: PASSED'
   else
      print '(A)', 'elisha_sv_fft_mpi: FAILED'
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

   subroutine check_ladder(label, nk, levels, periodic, prof)
   !< Solve on each level of a ladder, assert the discrete error (periodic), the residual and the continuum order.
   character(*), intent(in) :: label     !< Configuration label.
   integer(I4P), intent(in) :: nk        !< Cells per block along z (1: 2-D, null z).
   integer(I4P), intent(in) :: levels(:) !< Uniform refinement levels.
   logical,      intent(in) :: periodic  !< Periodic boundaries.
   integer(I4P), intent(in) :: prof      !< Source.
   real(R_P)                :: min_ord              !< Bound on the order.
   real(R_P)                :: e_disc(size(levels)) !< Error against the discrete solution, relative.
   real(R_P)                :: e_res(size(levels))  !< Discrete residual, relative.
   real(R_P)                :: e_cont(size(levels)) !< Max error against the continuum solution.
   integer(I4P)             :: n(size(levels))      !< Cells along x.
   integer(I4P)             :: unfilled(size(levels)) !< Phi cells the solve left.
   integer(I4P)             :: rank_diff(size(levels)) !< Values differing from rank 0's solo solve (-1: unchecked).
   real(R_P)                :: order                !< Observed order.
   logical                  :: ok                   !< Ladder pass flag.
   integer(I4P)             :: m                    !< Counter.
   character(16)            :: s(4)                 !< Formatted numbers.
   character(48)            :: rk                   !< Rank-invariance verdict.

   min_ord = MIN_ORDER ; if (prof == PROFILE_UNIFORM) min_ord = MIN_ORDER_SPHERE
   ok = .true.
   order = 0._R_P
   do m=1, size(levels)
      call solve_level(nk=nk, level=levels(m), periodic=periodic, prof=prof, n=n(m), e_disc=e_disc(m), e_res=e_res(m), &
                       e_cont=e_cont(m), unfilled=unfilled(m), rank_diff=rank_diff(m), stat=stat)
      if (stat /= ELISHA_OK) then
         ok = .false.
         exit
      endif
      if (prof == PROFILE_PERIODIC) then
         ok = ok .and. e_disc(m) <= TOL_DISCRETE .and. e_res(m) <= TOL_RESIDUAL .and. unfilled(m) == 0_I4P
      else
         ok = ok .and. e_res(m) <= TOL_RES_ISOLATED .and. unfilled(m) == 0_I4P
      endif
      ok = ok .and. rank_diff(m) <= 0_I4P
      if (c%rank == 0) print '(A,I0,A)', '   '//label//' N = ', n(m), ': |Phi - Phi_h| '//fmt(e_disc(m))// &
         ', residual '//fmt(e_res(m))//', |Phi - Phi_exact| '//fmt(e_cont(m))//', unfilled '//itoa(unfilled(m))
   enddo
   if (stat == ELISHA_OK .and. size(levels) > 1) then ! every level solved
      m = size(levels)
      if (prof == PROFILE_PERIODIC) then
         order = log(e_cont(m-1) / e_cont(m)) / log(real(n(m), R_P) / n(m-1))
         if (c%rank == 0) print '(A)', '   '//label//' order against the exact Phi, N = '//itoa(n(m-1))//' -> '// &
            itoa(n(m))//': '//fmtf(order)//' (min '//fmtf(min_ord)//')'
      else
         order = -slope(log(real(n, R_P)), log(e_cont))
         if (c%rank == 0) print '(A)', '   '//label//' order against the exact Phi, least squares over N = '// &
            itoa(n(1))//' to '//itoa(n(m))//': '//fmtf(order)//' (min '//fmtf(min_ord)//')'
      endif
      ok = ok .and. order >= min_ord
   endif
   if (c%rank == 0) then
      if (ok) then
         print '(A)', '   '//label//' PASS'
      else
         print '(A)', '   '//label//' FAIL'
      endif
   endif
   test_passed = test_passed .and. ok
   ! the summary row
   if (stat /= ELISHA_OK) then
      call add_row(label//' solve failed: '//elisha_status_name(stat)//'  FAIL')
      return
   endif
   if (prof == PROFILE_PERIODIC) then
      s(1) = fmt(maxval(e_disc))
   else
      s(1) = '-'
   endif
   s(2) = fmt(maxval(e_res)) ; s(3) = fmt(e_cont(size(levels)))
   if (size(levels) > 1) then
      s(4) = fmtf(order)
   else
      s(4) = '-'
   endif
   if (all(rank_diff < 0_I4P)) then
      rk = 'not checked'
   elseif (all(rank_diff <= 0_I4P)) then
      rk = 'bitwise'
   else
      rk = itoa(maxval(rank_diff))//' values differ'
   endif
   call add_row(label//' N '//itoa(n(1))//'-'//itoa(n(size(levels)))//' | disc '//trim(s(1))//' | resid '// &
                trim(s(2))//' | exact(finest) '//trim(s(3))//' | order '//trim(s(4))//' | unfilled '//          &
                itoa(sum(unfilled))//                                                                           &
                ' | np/1 '//trim(rk)//' | '//merge('PASS', 'FAIL', ok))
   endsubroutine check_ladder

   subroutine solve_level(nk, level, periodic, prof, n, e_disc, e_res, e_cont, unfilled, rank_diff, stat)
   !< Build the uniform mesh of one level (blocks in contiguous chunks over the ranks), solve, measure; on more than
   !< one rank re-solve on rank 0 alone and compare the transform arrays bit for bit; dump on request.
   integer(I4P), intent(in)  :: nk        !< Cells per block along z.
   integer(I4P), intent(in)  :: level     !< Uniform refinement level.
   logical,      intent(in)  :: periodic  !< Periodic boundaries.
   integer(I4P), intent(in)  :: prof      !< Source.
   integer(I4P), intent(out) :: n         !< Cells along x.
   real(R_P),    intent(out) :: e_disc    !< Error against the discrete solution, relative.
   real(R_P),    intent(out) :: e_res     !< Discrete residual, relative.
   real(R_P),    intent(out) :: e_cont    !< Max error against the continuum solution.
   integer(I4P), intent(out) :: unfilled  !< Phi cells the solve left.
   integer(I4P), intent(out) :: rank_diff !< Values differing from rank 0's solo solve (-1: unchecked).
   integer(I4P), intent(out) :: stat      !< Status.
   integer(I4P), allocatable :: all_c(:,:), my_c(:,:) !< Every block, this rank's blocks [4,*].
   real(R_P),    allocatable :: phi(:,:,:,:)  !< Potential (i, j, k, b).
   real(R_P),    allocatable :: f(:,:,:,:)    !< Source (i, j, k, b).
   real(R_P),    allocatable :: g(:,:,:)      !< Rank 0: the transform array.
   real(R_P),    allocatable :: g1(:,:,:)     !< Rank 0: the transform array of the solo solve.
   real(R_P),    allocatable :: phi1(:,:,:,:), f1(:,:,:,:) !< The solo solve's potential and source.
   real(R_P)                 :: xyz(3)        !< Cell centre.
   real(R_P)                 :: h(3)          !< Cell spacing.
   real(R_P)                 :: lap           !< L_h phi.
   real(R_P)                 :: e(4)          !< e_disc, e_res, e_cont, then the scales.
   real(R_P)                 :: scale(2)      !< Max |Phi_h|, max |4 pi G (rho - rho0)|.
   real(R_P)                 :: rho_ref       !< Density the solve removes (rho0 periodic, 0 isolated).
   logical                   :: is_iso        !< Isolated solve.
   logical                   :: active(3)     !< Active directions.
   logical                   :: need_g        !< The transform array is needed (dump or rank check).
   integer(I4P)              :: nijk(3)       !< Cells per block.
   integer(I4P)              :: m(3), n3(3)   !< Transform cells, global cells.
   integer(I4P)              :: nb            !< This rank's blocks.
   integer(I4P)              :: b, i, j, k, d !< Counters.
   integer(I4P)              :: o(3)          !< Unit offset along a direction.
   integer(I4P)              :: iv(2)         !< Unfilled count, rank_diff, reduced.
   integer(I4P)              :: ijk3(3)       !< A cell (a named actual argument).
   integer(I8P)              :: nu(1)         !< Unfilled count, summed.
   type(elisha_comm_object)  :: solo          !< Rank 0 alone.
   character(:), allocatable :: msg           !< Message.

   is_iso = prof /= PROFILE_PERIODIC
   rho_ref = RHO0 ; if (is_iso) rho_ref = 0._R_P
   nijk = [NB_SIDE, NB_SIDE, nk]
   active = [.true., .true., nk > 1]
   n = 0 ; e_disc = 0._R_P ; e_res = 0._R_P ; e_cont = 0._R_P ; unfilled = 0 ; rank_diff = -1
   call level_blocks(nk, level, c%rank, c%nranks, all_c, my_c)
   nb = size(my_c, dim=2)
   need_g = len_trim(dump_dir) > 0 .or. (rank_check .and. c%nranks > 1)
   call run_solve(cc=c, coords=my_c, nk=nk, level=level, periodic=periodic, prof=prof, need_g=need_g, phi=phi, f=f, &
                  h=h, n3=n3, m=m, g=g, stat=stat, msg=msg)
   if (stat /= ELISHA_OK) then
      if (c%rank == 0) print '(A)', '   solve failed: '//msg
      return
   endif
   n = n3(1)
   e = 0._R_P ; scale = 0._R_P ; unfilled = 0_I4P
   do b=1, nb
      do k=1-NGC, nijk(3)+NGC
         do j=1-NGC, nijk(2)+NGC
            do i=1-NGC, nijk(1)+NGC
               if (phi(i,j,k,b) == SENTINEL) then
                  unfilled = unfilled + 1_I4P
                  cycle
               endif
               ijk3 = [i, j, k] ! named: ifx -check arg_temp_created flags a constructor per call
               xyz = cell_centre(my_c(:,b), ijk3, nk, is_iso)
               if (is_iso) then ! no discrete closed form; the continuum error over every cell, ghosts included
                  e(3) = max(e(3), abs(phi(i,j,k,b) - phi_sphere(xyz, prof)))
                  scale(1) = 1._R_P
                  if (i < 1 .or. i > nijk(1) .or. j < 1 .or. j > nijk(2) .or. k < 1 .or. k > nijk(3)) cycle
               else
                  e(1)     = max(e(1), abs(phi(i,j,k,b) - phi_exact(xyz, active, h)))
                  scale(1) = max(scale(1), abs(phi_exact(xyz, active, h)))
                  if (i < 1 .or. i > nijk(1) .or. j < 1 .or. j > nijk(2) .or. k < 1 .or. k > nijk(3)) cycle
                  e(3) = max(e(3), abs(phi(i,j,k,b) - phi_exact(xyz, active)))
               endif
               lap = 0._R_P
               do d=1, 3
                  if (.not.active(d)) cycle
                  o = 0 ; o(d) = 1
                  lap = lap + (phi(i+o(1),j+o(2),k+o(3),b) - 2._R_P * phi(i,j,k,b) + &
                               phi(i-o(1),j-o(2),k-o(3),b)) / h(d)**2
               enddo
               e(2)     = max(e(2), abs(lap - FACTOR * (f(i,j,k,b) - rho_ref)))
               scale(2) = max(scale(2), abs(FACTOR * (f(i,j,k,b) - rho_ref)))
            enddo
         enddo
      enddo
   enddo
   e(4) = 0._R_P
   call c%max_real8(e(1:3), stat) ; if (stat /= ELISHA_OK) return
   call c%max_real8(scale, stat) ; if (stat /= ELISHA_OK) return
   nu = int(unfilled, I8P)
   call c%sum_i8(nu, stat) ; if (stat /= ELISHA_OK) return
   unfilled = int(nu(1), I4P)
   e_disc = e(1) / scale(1)
   e_res  = e(2) / scale(2)
   e_cont = e(3)
   if (len_trim(dump_dir) > 0 .and. c%rank == 0) &
      call dump(filename=trim(dump_dir)//'/phi-r'//itoa(merge(4_I4P, 8_I4P, nk == 1))//'-nk'//itoa(nk)//'-l'// &
                itoa(level)//'-p'//itoa(prof)//'.bin', n=m, f=g)
   if (rank_check .and. c%nranks > 1) then ! the same level on rank 0 alone, every block its own
      iv = 0
      if (c%rank == 0) then
         stat = ELISHA_OK
#ifndef ELISHA_NO_MPI
         call solo%init(MPI_COMM_SELF, stat, msg)
#endif
         if (stat == ELISHA_OK) call run_solve(cc=solo, coords=all_c, nk=nk, level=level, periodic=periodic, prof=prof, &
                                               need_g=.true., phi=phi1, f=f1, h=h, n3=n3, m=m, g=g1, stat=stat, msg=msg)
         if (stat /= ELISHA_OK) then
            print '(A)', '   the solo solve failed: '//msg
            iv(2) = huge(1_I4P)
         elseif (any(shape(g1) /= shape(g))) then
            iv(2) = size(g, kind=I4P)
         else
            iv(2) = bits_differ(g1, g)
         endif
         call solo%destroy(stat)
      endif
      call c%max_i4(iv, stat) ; if (stat /= ELISHA_OK) return
      rank_diff = iv(2)
   endif
   endsubroutine solve_level

   function bits_differ(g1, g2) result(nd)
   !< The values of two arrays of one shape whose bit patterns differ, by explicit loops: no array temporary (ifx puts
   !< them on the stack, and the transform arrays of the isolated 64^3 solve overflow its default 8 MB).
   real(R_P), intent(in) :: g1(:,:,:) !< First array.
   real(R_P), intent(in) :: g2(:,:,:) !< Second array, of the same shape.
   integer(I4P)          :: nd        !< Values that differ.
   integer(I4P)          :: i, j, k   !< Counters.

   nd = 0
   do k=1, size(g1, dim=3)
      do j=1, size(g1, dim=2)
         do i=1, size(g1, dim=1)
            if (transfer(g1(i,j,k), 0_I8P) /= transfer(g2(i,j,k), 0_I8P)) nd = nd + 1
         enddo
      enddo
   enddo
   endfunction bits_differ

   subroutine run_solve(cc, coords, nk, level, periodic, prof, need_g, phi, f, h, n3, m, g, stat, msg)
   !< On the context cc: the mesh of the given blocks, the setup, the source, the solve (phi set to the sentinel first),
   !< and on request the transform array (collective).
   type(elisha_comm_object),  intent(in)    :: cc          !< Communication context.
   integer(I4P),              intent(in)    :: coords(:,:) !< This rank's blocks [4,nb].
   integer(I4P),              intent(in)    :: nk          !< Cells per block along z.
   integer(I4P),              intent(in)    :: level       !< Level.
   logical,                   intent(in)    :: periodic    !< Periodic boundaries.
   integer(I4P),              intent(in)    :: prof        !< Source.
   logical,                   intent(in)    :: need_g      !< Return the transform array.
   real(R_P), allocatable,    intent(inout) :: phi(:,:,:,:) !< Potential.
   real(R_P), allocatable,    intent(inout) :: f(:,:,:,:)  !< Source.
   real(R_P),                 intent(out)   :: h(3)        !< Cell spacing of the solver.
   integer(I4P),              intent(out)   :: n3(3)       !< Global cells of the solver.
   integer(I4P),              intent(out)   :: m(3)        !< Transform cells of the solver.
   real(R_P), allocatable,    intent(inout) :: g(:,:,:)    !< The transform array (rank 0 of cc).
   integer(I4P),              intent(out)   :: stat        !< Status.
   character(:), allocatable, intent(out)   :: msg         !< Message.
   type(elisha_mesh_object)            :: mesh   !< Mesh.
   class(elisha_spectral_solver_object), allocatable :: solver !< Solver.
   real(R_P),    allocatable :: buf(:)           !< Packed source.
   real(R_P)                 :: lb(3), emin(3)   !< Domain lengths, lower corner.
   real(R_P)                 :: dxc(3)           !< Cell spacing (ADAM's field%dxyz).
   logical                   :: is_iso, cube     !< Isolated solve, the cube domain.
   logical                   :: null(3)          !< Null axes.
   integer(I4P)              :: nijk(3), nb, b, i, j, k, p !< Cells per block, blocks, counters.
   integer(I4P)              :: ijk3(3), nblk1(3) !< A cell, the blocks at level 0 (named actual arguments).
   real(R_P)                 :: dx0(3), xyz(3)   !< Spacing at level 0, a cell centre.
   logical                   :: per3(3), act3(3) !< Periodic axes, active axes.

   call new_fft(solver)
   is_iso = prof /= PROFILE_PERIODIC
   cube = is_iso
   nijk = [NB_SIDE, NB_SIDE, nk]
   null = [.false., .false., nk == 1]
   call domain(cube, lb, emin)
   nb = size(coords, dim=2)
   ! named arguments, no expressions: ifx -check arg_temp_created flags every temporary
   nblk1 = 1_I4P ; dx0 = lb / real(nijk, R_P) ; per3 = merge(.not.null, [.false., .false., .false.], periodic)
   act3 = [.true., .true., nk > 1]
   call mesh%init(nijk, NGC, nblk1, dx0, emin, per3, null, coords, stat, msg)
   if (stat /= ELISHA_OK) return
   call solver%setup(mesh, cc, stat, msg, isolated=is_iso)
   if (stat /= ELISHA_OK) return
   dxc = block_dx(lb, level, nk) / real(nijk, R_P)
   if (allocated(f)) deallocate(f)
   if (allocated(phi)) deallocate(phi)
   allocate(f(nijk(1),nijk(2),nijk(3),max(nb,1)), buf(max(nb*product(nijk),1)))
   allocate(phi(1-NGC:nijk(1)+NGC,1-NGC:nijk(2)+NGC,1-NGC:nijk(3)+NGC,1:max(nb,1)))
   f = 0._R_P ; buf = 0._R_P
   p = 0
   do b=1, nb
      do k=1, nijk(3)
         do j=1, nijk(2)
            do i=1, nijk(1)
               ijk3 = [i, j, k]
               xyz = cell_centre(coords(:,b), ijk3, nk, cube)
               if (is_iso) then
                  f(i,j,k,b) = rho_sphere(xyz, prof, dxc)
               else
                  f(i,j,k,b) = rho(xyz, act3)
               endif
               p = p + 1 ; buf(p) = f(i,j,k,b)
            enddo
         enddo
      enddo
   enddo
   phi = SENTINEL
   call solver%solve_packed(mesh, buf, FACTOR, phi, stat, msg)
   if (stat /= ELISHA_OK) return
   h = solver%dxyz ; n3 = solver%n ; m = solver%m
   if (need_g) call solver%global_potential(g, stat)
   call solver%destroy
   call mesh%destroy
   endsubroutine run_solve

   subroutine check_refusal(label, nk, level, periodic, isolated, cube, expected)
   !< A configuration the solver cannot solve: the setup must return the expected code, on every rank.
   character(*), intent(in) :: label    !< Configuration label.
   integer(I4P), intent(in) :: nk       !< Cells per block along z.
   integer(I4P), intent(in) :: level    !< Level.
   logical,      intent(in) :: periodic !< Periodic boundaries.
   logical,      intent(in) :: isolated !< Isolated solve.
   logical,      intent(in) :: cube     !< The cube [-1,1]^3 (else [0,1] x [0,2] x [0,3]).
   integer(I4P), intent(in) :: expected !< Expected status.
   type(elisha_mesh_object)            :: mesh   !< Mesh.
   class(elisha_spectral_solver_object), allocatable :: solver !< Solver.
   integer(I4P), allocatable :: all_c(:,:), my_c(:,:) !< Every block, this rank's blocks.
   real(R_P)                 :: lb(3), emin(3)   !< Domain lengths, lower corner.
   logical                   :: null(3)          !< Null axes.
   integer(I4P)              :: nijk(3), st(1), lo(1), hi(1) !< Cells per block, status and its range over the ranks.
   integer(I4P)              :: nblk1(3)         !< The blocks at level 0 (a named actual argument).
   real(R_P)                 :: dx0(3)           !< Spacing at level 0.
   logical                   :: per3(3)          !< Periodic axes.
   logical                   :: ok               !< Pass flag.
   character(:), allocatable :: msg              !< Message.

   call new_fft(solver)
   nijk = [NB_SIDE, NB_SIDE, nk]
   null = [.false., .false., nk == 1]
   call domain(cube, lb, emin)
   call level_blocks(nk, level, c%rank, c%nranks, all_c, my_c)
   nblk1 = 1_I4P ; dx0 = lb / real(nijk, R_P) ; per3 = merge(.not.null, [.false., .false., .false.], periodic)
   call mesh%init(nijk, NGC, nblk1, dx0, emin, per3, null, my_c, st(1), msg)
   if (st(1) == ELISHA_OK) call solver%setup(mesh, c, st(1), msg, isolated=isolated)
   lo = st ; hi = st
   call c%min_i4(lo, stat) ; call c%max_i4(hi, stat)
   ok = lo(1) == expected .and. hi(1) == expected
   if (c%rank == 0) then
      print '(A)', '   refusal, '//label//': '//elisha_status_name(st(1))//' (expected '// &
                   elisha_status_name(expected)//')  '//merge('PASS', 'FAIL', ok)
      if (allocated(msg)) print '(A)', '      '//msg
   endif
   call add_row('refusal, '//label//': '//elisha_status_name(st(1))//'  '//merge('PASS', 'FAIL', ok))
   test_passed = test_passed .and. ok
   call solver%destroy
   call mesh%destroy
   endsubroutine check_refusal

   subroutine level_blocks(nk, level, rank, nranks, all_c, my_c)
   !< Every block of a uniform level (x fastest) and this rank's contiguous share.
   integer(I4P),              intent(in)    :: nk          !< Cells per block along z (1: null z).
   integer(I4P),              intent(in)    :: level       !< Level.
   integer(I4P),              intent(in)    :: rank        !< This rank.
   integer(I4P),              intent(in)    :: nranks      !< Ranks.
   integer(I4P), allocatable, intent(inout) :: all_c(:,:)  !< Every block [4,ntot].
   integer(I4P), allocatable, intent(inout) :: my_c(:,:)   !< This rank's blocks [4,nb].
   integer(I4P)                             :: nbl(3), ntot, gb, lo, hi !< Blocks per axis, total, counters.

   nbl = [2_I4P**level, 2_I4P**level, merge(1_I4P, 2_I4P**level, nk == 1)]
   ntot = product(nbl)
   if (allocated(all_c)) deallocate(all_c)
   if (allocated(my_c)) deallocate(my_c)
   allocate(all_c(4,ntot))
   do gb=0, ntot-1
      all_c(:,gb+1) = [mod(gb, nbl(1)), mod(gb / nbl(1), nbl(2)), gb / (nbl(1) * nbl(2)), level]
   enddo
   lo = int((int(rank, I8P) * ntot) / nranks, I4P)
   hi = int((int(rank + 1, I8P) * ntot) / nranks, I4P)
   my_c = all_c(:,lo+1:hi)
   endsubroutine level_blocks

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

   pure function block_dx(lb, level, nk) result(bdx)
   !< ADAM's block size at a level: the domain over 2^level along an active axis (the whole length along a null one).
   real(R_P),    intent(in) :: lb(3)  !< Domain lengths.
   integer(I4P), intent(in) :: level  !< Level.
   integer(I4P), intent(in) :: nk     !< Cells per block along z (1: null z).
   real(R_P)                :: bdx(3) !< Block size.

   bdx = lb / 2_I4P**level
   if (nk == 1) bdx(3) = lb(3)
   endfunction block_dx

   pure function cell_centre(c4, ijk, nk, cube) result(xyz)
   !< The centre of cell ijk (ghosts included) of block c4, in ADAM's arithmetic (grid_object%cell_xyz): the block's
   !< corner, minus half a cell, plus ijk cells.
   integer(I4P), intent(in) :: c4(4)  !< Block (i, j, k, level).
   integer(I4P), intent(in) :: ijk(3) !< Cell.
   integer(I4P), intent(in) :: nk     !< Cells per block along z.
   logical,      intent(in) :: cube   !< The cube domain.
   real(R_P)                :: xyz(3) !< Centre.
   real(R_P)                :: lb(3), emin(3), bdx(3), cdx(3), bmin(3) !< Domain, block size, cell size, block corner.
   integer(I4P)             :: nijk(3) !< Cells per block.

   call domain(cube, lb, emin)
   nijk = [NB_SIDE, NB_SIDE, nk]
   bdx = block_dx(lb, c4(4), nk)
   cdx = bdx / nijk
   bmin = emin + c4(1:3) * bdx
   xyz = bmin - bdx / nijk * 0.5_R_P + ijk * cdx
   endfunction cell_centre

   subroutine dump(filename, n, f)
   !< Write rank 0's transform array: the 3 sizes (int32), then the values (float64, x fastest).
   character(*), intent(in) :: filename !< File name.
   integer(I4P), intent(in) :: n(3)     !< Sizes.
   real(R_P),    intent(in) :: f(:,:,:) !< Transform array.
   integer                  :: u, ios   !< File unit, I/O status.

   open(newunit=u, file=filename, access='stream', form='unformatted', action='write', status='replace', iostat=ios)
   if (ios /= 0) then
      print '(A)', '   dump: cannot open '//filename
      test_passed = .false.
      return
   endif
   write(u) n
   write(u) f
   close(u)
   endsubroutine dump

   pure function wave(m, active) result(kv)
   !< Wave vector of mode m (1 or 2); no wave along an inactive direction.
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
   !< The periodic source density at a point.
   real(R_P), intent(in) :: xyz(3)       !< Point.
   logical,   intent(in) :: active(3)    !< Active directions.
   real(R_P)             :: r            !< Density.
   real(R_P)             :: k1(3), k2(3) !< Wave vectors.

   k1 = wave(1_I4P, active) ; k2 = wave(2_I4P, active)
   r = RHO0 + A1 * sin(k1(1) * xyz(1)) * cos(k1(2) * xyz(2)) * cos(k1(3) * xyz(3)) + A2 * cos(dot_product(k2, xyz))
   endfunction rho

   pure function phi_exact(xyz, active, h) result(p)
   !< The exact periodic potential: continuum (eigenvalue -|k|^2), or discrete with the spacing h (eigenvalue of L_h).
   real(R_P), intent(in)           :: xyz(3)       !< Point.
   logical,   intent(in)           :: active(3)    !< Active directions.
   real(R_P), intent(in), optional :: h(3)         !< Cell spacing: the discrete solution.
   real(R_P)                       :: p            !< Potential.
   real(R_P)                       :: k1(3), k2(3) !< Wave vectors.
   real(R_P)                       :: lam1, lam2   !< Eigenvalues.

   k1 = wave(1_I4P, active) ; k2 = wave(2_I4P, active)
   if (present(h)) then
      lam1 = eigenvalue(k1, active, h) ; lam2 = eigenvalue(k2, active, h)
   else
      lam1 = -dot_product(k1, k1) ; lam2 = -dot_product(k2, k2)
   endif
   p = FACTOR * (A1 * sin(k1(1) * xyz(1)) * cos(k1(2) * xyz(2)) * cos(k1(3) * xyz(3)) / lam1 + &
                 A2 * cos(dot_product(k2, xyz)) / lam2)
   endfunction phi_exact

   pure function eigenvalue(kv, active, h) result(lam)
   !< Eigenvalue of L_h for a mode of wave vector kv.
   real(R_P), intent(in) :: kv(3)     !< Wave vector.
   logical,   intent(in) :: active(3) !< Active directions.
   real(R_P), intent(in) :: h(3)      !< Cell spacing.
   real(R_P)             :: lam       !< Eigenvalue.
   integer(I4P)          :: d         !< Counter.

   lam = 0._R_P
   do d=1, 3
      if (active(d)) lam = lam + (2._R_P * cos(kv(d) * h(d)) - 2._R_P) / h(d)**2
   enddo
   endfunction eigenvalue

   pure function slope(x, y) result(sl)
   !< Least-squares slope of y against x.
   real(R_P), intent(in) :: x(:) !< Abscissae.
   real(R_P), intent(in) :: y(:) !< Ordinates.
   real(R_P)             :: sl   !< Slope.

   sl = sum((x - sum(x) / size(x)) * (y - sum(y) / size(y))) / sum((x - sum(x) / size(x))**2)
   endfunction slope

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
endprogram elisha_sv_fft_mpi
