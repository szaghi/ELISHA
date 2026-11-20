!< ELISHA device test: ADAM's device Poisson solvers (OpenACC + cuFFT) solve what they claim and equal their host twins.
program elisha_test_adam_device_oracle_mpi
!< ELISHA device test: ADAM's device Poisson solvers (OpenACC + cuFFT) solve what they claim and equal their host twins.
!<
!< The device solvers are `adam_fnl_poisson_fft_object` and `adam_fnl_poisson_mg_object` (ADAM 669d0bdd verbatim, in
!< `src/tests/device/`, built only by the fobos mode `tests-nvf-oac`: `-acc -cudalib=cufft -D_CUFFT`); the host ones
!< the verbatim `adam_poisson_fft_object` and `adam_poisson_mg_object` they extend; all over the test shims
!< `adam_grid_object`, `adam_field_object`, `adam_mpih_global`. Every case solves one source with the host solver and
!< with its device twin, in one program, the blocks (leaves) dealt round-robin to the ranks:
!< - FFT, on uniform grids (the distributed slab path: each rank's GPU transforms its planes, the transpose on the
!<   device buffers by GPU-aware `MPI_ALLTOALLV`): periodic cubic 3-D at level 0, periodic non-cubic at level 1,
!<   periodic 2-D (null z), and isolated 3-D (the kernel's spectrum kept on the device). Checked: by meaning, the
!<   7-point (5-point) Laplacian of the device `phi`, read through the ghost layers, equals `factor (f - mean f)`
!<   (isolated: `factor f`) to a relative max-norm of `TOL_RES` (`TOL_RES_ISO`); device against host,
!<   `max |phi_dev - phi_host| / max |phi_host|` over the interior at most `TOL_DEV_FFT`.
!< - Multigrid (the device FCG and V-cycle; the root on every rank's cuFFT, redundant): periodic uniform with two levels
!<   inside the blocks (root 64 cells), 2:1 AMR with a corner block refined (its children meet level-0 leaves across
!<   the wrap), 2:1 AMR on 3^3 with the centre block refined (root 216 cells), uniform with the default root (one
!<   V-cycle is the exact solve), and ghost-Dirichlet AMR (the root on the odd extension). Both solvers get the same
!<   arguments (the same hierarchy: `lbot` and `lmax` must agree). Checked on the device solver by meaning, as
!<   `elisha_test_adam_mg_oracle_mpi` does on the host one: its FCG residual at most its stop tolerance, at least one
!<   iteration, the residual recomputed through its own `apply_a` at most twice the stop tolerance, its
!<   `symmetry_defect` at most `TOL_SYMMETRY`; device against host: `apply_a` of one fixed vector,
!<   `max |A_dev x - A_host x| / max |A_host x|` at most `TOL_DEV_OP` (the same rows summed in the same order: only
!<   FMA contraction differs); the potentials, `max |phi_dev - phi_host| / max |phi_host|`, at most `TOL_DEV_EXACT`
!<   when the solve is one exact V-cycle (one iteration on both), at most `TOL_DEV_FCG_FACTOR` times the stop tolerance
!<   otherwise (each potential is within the tolerance of the discrete solution; equal iteration counts give round-off,
!<   printed). ADAM (flail.md): the device results equal the host's to round-off, the dots summed in another order and
!<   products contracted into FMAs.
!< Rank 0 prints a line per case and one summary line, `elisha_test_adam_device_oracle_mpi: ok (...)` or `FAILED (...)`,
!< with the worst device-host differences. Each rank takes the GPU of its node-local rank modulo the node's GPUs; no
!< GPU fails the test (it must not pass on a host fallback). GPU-aware MPI is needed on more than one rank (HPC-X).
!<
!< `-DELISHA_DEVICE_ORACLE_DRYRUN` builds the program with the host solvers in place of the device ones (no OpenACC,
!< no cuFFT): a check, on any machine, of the program's own logic (every device-host difference is then zero).
!< The FP traps of a debug build are off around `MPI_Init` only (hwloc raises IEEE flags inside it).
#ifdef ELISHA_DEVICE_ORACLE_DRYRUN
#define FFT_DEV_T poisson_fft_object
#define MG_DEV_T poisson_mg_object
#else
#define FFT_DEV_T poisson_fft_fnl_object
#define MG_DEV_T poisson_mg_fnl_object
#endif
use :: mpi
use, intrinsic :: ieee_exceptions, only : ieee_usual, ieee_get_halting_mode, ieee_set_halting_mode
#ifdef _OPENACC
use :: openacc
#endif
use :: penf, only : I4P, I8P, R8P
use :: adam_grid_object, only : grid_object
use :: adam_field_object, only : field_object
use :: adam_mpih_global, only : mpih
use :: adam_poisson_fft_object, only : poisson_fft_object
use :: adam_poisson_mg_object, only : poisson_mg_object
#ifndef ELISHA_DEVICE_ORACLE_DRYRUN
use :: adam_fnl_poisson_fft_object, only : poisson_fft_fnl_object
use :: adam_fnl_poisson_mg_object, only : poisson_mg_fnl_object
#endif
implicit none
real(R8P),    parameter :: TOL_RES=1.e-12_R8P         !< FFT: relative max-norm residual allowed (periodic).
real(R8P),    parameter :: TOL_RES_ISO=1.e-11_R8P     !< FFT: relative max-norm residual allowed (isolated).
real(R8P),    parameter :: TOL_DEV_FFT=1.e-12_R8P     !< FFT: max |phi_dev - phi_host| / max |phi_host|.
real(R8P),    parameter :: TOL_MG=1.e-11_R8P          !< MG: FCG tolerance requested.
integer(I4P), parameter :: MAX_IT=100_I4P             !< MG: FCG iterations allowed.
real(R8P),    parameter :: TOL_SYMMETRY=1.e-12_R8P    !< MG: bound on the device operator's symmetry defect.
real(R8P),    parameter :: TOL_DEV_OP=1.e-13_R8P      !< MG: max |A_dev x - A_host x| / max |A_host x|.
real(R8P),    parameter :: TOL_DEV_EXACT=1.e-12_R8P   !< MG, one exact V-cycle: max |phi_dev - phi_host| / max |phi_host|.
real(R8P),    parameter :: TOL_DEV_FCG_FACTOR=1.e3_R8P !< MG, iterated: the bound above over the stop tolerance.
real(R8P),    parameter :: FACTOR=2.5_R8P             !< Source factor.
real(R8P),    parameter :: PI=acos(-1._R8P)           !< Pi.
real(R8P),    parameter :: C0=0.7_R8P                 !< The constant of the source.
integer(I4P), parameter :: NGHOST=2_I4P               !< Ghost layers.
integer(I4P)            :: failures, nfail            !< Failed checks, on this rank and over the ranks.
integer                 :: ierr                       !< MPI error status.
integer                 :: ndev                       !< GPUs of this rank's node (the most over the ranks).
logical                 :: halting(3)                 !< The FP halting modes of the build (debug: on).
real(R8P)               :: worst(4)                   !< Worst max rel |dev-host|: FFT, MG A x, MG exact, MG FCG.

failures = 0 ; worst = 0._R8P ; ndev = 0
! MPI_Init runs C code (hwloc's XML backend) that raises IEEE flags: a debug build's FP traps (-ffpe-trap) would
! kill it there, so they are off around the call only (Open MPI 5.0.11, hwloc 2.7.1)
call ieee_get_halting_mode(ieee_usual, halting)
call ieee_set_halting_mode(ieee_usual, .false.)
call MPI_INIT(ierr)
call ieee_set_halting_mode(ieee_usual, halting)
call mpih%initialize
call select_device
if (failures == 0) then
   if (mpih%myrank == 0) print '(A,I0,A,I0,A)', 'elisha_test_adam_device_oracle_mpi: np ', mpih%procs_number, &
                                               ', GPUs per node ', ndev, &
#ifdef ELISHA_DEVICE_ORACLE_DRYRUN
                                               ' (DRY RUN: host solvers in place of the device ones)'
#else
                                               ' (device: ADAM FNL, OpenACC + cuFFT)'
#endif
   ! FFT, uniform grids
   call run_fft(label='FFT periodic cubic, lvl 0', nijk=[8_I4P, 8_I4P, 8_I4P], nblk0=[2_I4P, 2_I4P, 2_I4P], &
                level=0_I4P, dx0=[1._R8P, 1._R8P, 1._R8P] / 16._R8P, emin=[0._R8P, 0._R8P, 0._R8P],     &
                null=[.false., .false., .false.], isolated=.false.)
   call run_fft(label='FFT periodic non-cubic, l1', nijk=[8_I4P, 4_I4P, 6_I4P], nblk0=[1_I4P, 2_I4P, 1_I4P], &
                level=1_I4P, dx0=[0.1_R8P, 0.07_R8P, 0.13_R8P], emin=[-0.3_R8P, 0.2_R8P, 1._R8P],           &
                null=[.false., .false., .false.], isolated=.false.)
   call run_fft(label='FFT periodic 2-D (null z) ', nijk=[8_I4P, 8_I4P, 1_I4P], nblk0=[2_I4P, 3_I4P, 1_I4P], &
                level=0_I4P, dx0=[0.05_R8P, 0.08_R8P, 1._R8P], emin=[0._R8P, 0._R8P, 0._R8P],               &
                null=[.false., .false., .true.], isolated=.false.)
   call run_fft(label='FFT isolated cubic, lvl 0 ', nijk=[8_I4P, 8_I4P, 8_I4P], nblk0=[2_I4P, 2_I4P, 2_I4P], &
                level=0_I4P, dx0=[1._R8P, 1._R8P, 1._R8P] / 16._R8P, emin=[-0.5_R8P, -0.5_R8P, -0.5_R8P],  &
                null=[.false., .false., .false.], isolated=.true.)
   ! multigrid, periodic uniform and 2:1 AMR, and ghost-Dirichlet AMR
   call run_mg(label='MG (a) uniform, root 64    ', nblk0=[2_I4P, 2_I4P, 2_I4P], dx0=[1._R8P, 1._R8P, 1._R8P] / 16._R8P, &
               amr=.false., refine=[0_I4P, 0_I4P, 0_I4P], root_cells=64_I4P, periodic=.true., exact=.false.)
   call run_mg(label='MG (b) AMR 2^3, corner     ', nblk0=[2_I4P, 2_I4P, 2_I4P], dx0=[1._R8P, 1._R8P, 1._R8P] / 16._R8P, &
               amr=.true., refine=[0_I4P, 0_I4P, 0_I4P], root_cells=0_I4P, periodic=.true., exact=.false.)
   call run_mg(label='MG (b) AMR 3^3, centre, 216', nblk0=[3_I4P, 3_I4P, 3_I4P], dx0=[1._R8P, 1._R8P, 1._R8P] / 24._R8P, &
               amr=.true., refine=[1_I4P, 1_I4P, 1_I4P], root_cells=216_I4P, periodic=.true., exact=.false.)
   call run_mg(label='MG (c) uniform, exact root ', nblk0=[2_I4P, 2_I4P, 2_I4P], dx0=[1._R8P, 1._R8P, 1._R8P] / 16._R8P, &
               amr=.false., refine=[0_I4P, 0_I4P, 0_I4P], root_cells=0_I4P, periodic=.true., exact=.true.)
   call run_mg(label='MG (d) AMR ghost-Dirichlet ', nblk0=[2_I4P, 2_I4P, 2_I4P], dx0=[1._R8P, 1._R8P, 1._R8P] / 16._R8P, &
               amr=.true., refine=[1_I4P, 0_I4P, 1_I4P], root_cells=0_I4P, periodic=.false., exact=.false.)
endif
nfail = failures
call MPI_ALLREDUCE(MPI_IN_PLACE, nfail, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)
if (mpih%myrank == 0) then
   if (nfail == 0) then
      print '(A,I0,A,I0,A,4(A,ES9.2))', 'elisha_test_adam_device_oracle_mpi: ok (', mpih%procs_number, ' rank(s), ', &
         ndev, ' GPU(s) per node)', '; max rel |dev-host|: FFT ', worst(1), ', MG A x ', worst(2), ', MG exact ', &
         worst(3), ', MG FCG ', worst(4)
   else
      print '(A,I0,A,I0,A,I0,A,4(A,ES9.2))', 'elisha_test_adam_device_oracle_mpi: FAILED (', nfail, ' check(s), ', &
         mpih%procs_number, ' rank(s), ', ndev, ' GPU(s) per node)', '; max rel |dev-host|: FFT ', worst(1), &
         ', MG A x ', worst(2), ', MG exact ', worst(3), ', MG FCG ', worst(4)
   endif
endif
call MPI_FINALIZE(ierr)
if (nfail > 0) error stop 1

contains
   subroutine select_device
   !< Each rank on the GPU of its node-local rank modulo the node's GPUs, initialised; no GPU is a failure.
#ifdef _OPENACC
   integer :: local_comm, local_rank, nd, ierr_ !< Node communicator, node-local rank, GPUs, MPI error status.

   call MPI_COMM_SPLIT_TYPE(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, local_comm, ierr_)
   call MPI_COMM_RANK(local_comm, local_rank, ierr_)
   call MPI_COMM_FREE(local_comm, ierr_)
   nd = acc_get_num_devices(acc_device_nvidia)
   if (nd > 0) then
      call acc_set_device_num(mod(local_rank, nd), acc_device_nvidia)
      call acc_init(acc_device_nvidia)
   else
      call fail('no NVIDIA GPU visible to this rank (the device solvers would not run on the device)')
   endif
   call MPI_ALLREDUCE(nd, ndev, 1, MPI_INTEGER, MPI_MAX, MPI_COMM_WORLD, ierr_)
#else
   ndev = 0
#ifndef ELISHA_DEVICE_ORACLE_DRYRUN
   call fail('built without OpenACC (-acc): the device solvers would run on the host')
#endif
#endif
   endsubroutine select_device

   subroutine deal_blocks(nleaf, leaves, coords)
   !< This rank's share of the leaves, round-robin: leaf g (0-based) to rank mod(g, ranks).
   integer(I4P),              intent(in)  :: nleaf          !< Leaves.
   integer(I4P),              intent(in)  :: leaves(:,:)    !< Every leaf (i, j, k, level) [4,nleaf].
   integer(I4P), allocatable, intent(out) :: coords(:,:)    !< This rank's leaves [4,nb].
   integer(I4P)                           :: g, nb          !< Counter, leaves of this rank.

   nb = 0
   do g=0, nleaf-1
      if (mod(g, mpih%procs_number) == mpih%myrank) nb = nb + 1
   enddo
   allocate(coords(4,nb))
   nb = 0
   do g=0, nleaf-1
      if (mod(g, mpih%procs_number) /= mpih%myrank) cycle
      nb = nb + 1
      coords(:,nb) = leaves(:,g+1)
   enddo
   endsubroutine deal_blocks

   subroutine run_fft(label, nijk, nblk0, level, dx0, emin, null, isolated)
   !< One uniform grid: the host FFT solve and the device one; the device residual and the device-host difference.
   character(*), intent(in)  :: label    !< Case label.
   integer(I4P), intent(in)  :: nijk(3)  !< Cells per block per axis.
   integer(I4P), intent(in)  :: nblk0(3) !< Blocks per axis at level 0.
   integer(I4P), intent(in)  :: level    !< Level of every block.
   real(R8P),    intent(in)  :: dx0(3)   !< Cell spacing at level 0.
   real(R8P),    intent(in)  :: emin(3)  !< Domain lower corner.
   logical,      intent(in)  :: null(3)  !< Null axes.
   logical,      intent(in)  :: isolated !< Isolated boundaries (else periodic).
   type(grid_object)         :: grid     !< ADAM's grid (shim).
   type(field_object)        :: field    !< ADAM's field (shim).
   type(poisson_fft_object)  :: host     !< ADAM's FFT solver, host.
   type(FFT_DEV_T)           :: dev      !< ADAM's FFT solver, device.
   integer(I4P), allocatable :: leaves(:,:), coords(:,:) !< Every block, this rank's blocks [4,*].
   real(R8P),    allocatable :: q(:,:,:,:,:), phi_h(:,:,:,:), phi_d(:,:,:,:) !< Source, potentials.
   integer(I4P)              :: nbl(3), ncell(3), nb, g, b, i, j, k, gi, gj, gk, same !< Blocks, cells, counters.
   real(R8P)                 :: dx(3), mean, res_h, res_d, cmp(2) !< Spacing, mean, residuals, difference and scale.
   integer                   :: ierr_    !< MPI error status.

   call grid%set(nijk=nijk, ngc=NGHOST, nblk0=nblk0, dx0=dx0, emin=emin, periodic=.not.(null .or. isolated), &
                 null=null, maxlevel=level)
   nbl = merge(nblk0, nblk0 * 2_I4P**level, null)
   ncell = nbl * nijk
   dx = grid%cell_dxyz(:,level)
   allocate(leaves(4,product(nbl)))
   do g=0, product(nbl)-1
      leaves(:,g+1) = [mod(g, nbl(1)), mod(g / nbl(1), nbl(2)), g / (nbl(1) * nbl(2)), level]
   enddo
   call deal_blocks(product(nbl), leaves, coords)
   nb = size(coords, dim=2)
   call field%set(coords)
   allocate(q(1:1,1-NGHOST:nijk(1)+NGHOST,1-NGHOST:nijk(2)+NGHOST,1-NGHOST:nijk(3)+NGHOST,1:max(nb,1)))
   allocate(phi_h(1-NGHOST:nijk(1)+NGHOST,1-NGHOST:nijk(2)+NGHOST,1-NGHOST:nijk(3)+NGHOST,1:max(nb,1)))
   allocate(phi_d, mold=phi_h)
   q = 0._R8P ; phi_h = 0._R8P ; phi_d = 0._R8P
   ! the mean as C0 plus the mean deviation from it (the sum's round-off below the residual measured)
   mean = 0._R8P
   do b=1, nb
      do k=1, nijk(3)
         do j=1, nijk(2)
            do i=1, nijk(1)
               gi = coords(1,b) * nijk(1) + i - 1 ; gj = coords(2,b) * nijk(2) + j - 1
               gk = coords(3,b) * nijk(3) + k - 1
               q(1,i,j,k,b) = fft_source(gi, gj, gk, ncell)
               mean = mean + (q(1,i,j,k,b) - C0)
            enddo
         enddo
      enddo
   enddo
   call MPI_ALLREDUCE(MPI_IN_PLACE, mean, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr_)
   mean = C0 + mean / real(product(ncell), R8P)
   if (isolated) mean = 0._R8P ! isolated: L_h phi = factor f, phi -> 0 at infinity
   call host%initialize(grid=grid, field=field, isolated=isolated)
   call host%solve(field=field, q=q, iv=1_I4P, factor=FACTOR, phi=phi_h)
   call dev%initialize(grid=grid, field=field, isolated=isolated)
   call dev%solve(field=field, q=q, iv=1_I4P, factor=FACTOR, phi=phi_d)
   res_h = fft_residual(phi_h, q, mean, dx, null, nijk, nb)
   res_d = fft_residual(phi_d, q, mean, dx, null, nijk, nb)
   cmp = 0._R8P ; same = 0
   do b=1, nb
      do k=1, nijk(3)
         do j=1, nijk(2)
            do i=1, nijk(1)
               cmp(1) = max(cmp(1), abs(phi_d(i,j,k,b) - phi_h(i,j,k,b)))
               cmp(2) = max(cmp(2), abs(phi_h(i,j,k,b)))
               if (transfer(phi_d(i,j,k,b), 0_I8P) == transfer(phi_h(i,j,k,b), 0_I8P)) same = same + 1
            enddo
         enddo
      enddo
   enddo
   call MPI_ALLREDUCE(MPI_IN_PLACE, cmp, 2, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, ierr_)
   call MPI_ALLREDUCE(MPI_IN_PLACE, same, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr_)
   worst(1) = max(worst(1), cmp(1) / cmp(2))
   if (mpih%myrank == 0) print '(A,3(I0,1X),A,ES9.2,A,ES9.2,A,ES9.2,A,I0,A,I0)', &
      'elisha_test_adam_device_oracle_mpi: '//label//' cells ', ncell, 'residual dev ', res_d, ' host ', res_h, &
      ' | max rel |dev-host| ', cmp(1) / cmp(2), ' bitwise-equal ', same, ' of ', product(ncell)
   if (.not.(res_d <= merge(TOL_RES_ISO, TOL_RES, isolated))) call fail(label//': device residual above bound')
   if (.not.(cmp(1) <= TOL_DEV_FFT * cmp(2))) call fail(label//': device potential differs from the host one')
   call dev%destroy
   call host%destroy

   endsubroutine run_fft

   function fft_residual(phi, q, mean, dx, null, nijk, nb) result(r)
   !< max |L_h phi - factor (f - mean)| / max |factor (f - mean)| over the interior of every rank, through the ghosts.
   real(R8P),    intent(in) :: phi(1-NGHOST:,1-NGHOST:,1-NGHOST:,1:)    !< Potential.
   real(R8P),    intent(in) :: q(1:,1-NGHOST:,1-NGHOST:,1-NGHOST:,1:)   !< Source [1,i,j,k,nb].
   real(R8P),    intent(in) :: mean    !< The source's mean removed (0: none).
   real(R8P),    intent(in) :: dx(3)   !< Cell spacing.
   logical,      intent(in) :: null(3) !< Null axes.
   integer(I4P), intent(in) :: nijk(3) !< Cells per block per axis.
   integer(I4P), intent(in) :: nb      !< This rank's blocks.
   real(R8P)                :: r       !< Relative residual.
   real(R8P)                :: lap, s(2) !< Laplacian, residual and scale.
   integer(I4P)             :: b, i, j, k !< Counters.
   integer                  :: ierr_   !< MPI error status.

   s = 0._R8P
   do b=1, nb
      do k=1, nijk(3)
         do j=1, nijk(2)
            do i=1, nijk(1)
               lap = (phi(i+1,j,k,b) - 2._R8P * phi(i,j,k,b) + phi(i-1,j,k,b)) / dx(1)**2 + &
                     (phi(i,j+1,k,b) - 2._R8P * phi(i,j,k,b) + phi(i,j-1,k,b)) / dx(2)**2
               if (.not.null(3)) lap = lap + (phi(i,j,k+1,b) - 2._R8P * phi(i,j,k,b) + phi(i,j,k-1,b)) / dx(3)**2
               s(1) = max(s(1), abs(lap - FACTOR * (q(1,i,j,k,b) - mean)))
               s(2) = max(s(2), abs(FACTOR * (q(1,i,j,k,b) - mean)))
            enddo
         enddo
      enddo
   enddo
   call MPI_ALLREDUCE(MPI_IN_PLACE, s, 2, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, ierr_)
   r = s(1) / s(2)
   endfunction fft_residual

   subroutine run_mg(label, nblk0, dx0, amr, refine, root_cells, periodic, exact)
   !< One grid of 8^3-cell blocks, `nblk0` at level 0, block `refine` refined to level 1 when `amr`: the host multigrid
   !< solve and the device one; the device checks by meaning and the device-host differences.
   character(*), intent(in)  :: label      !< Case label.
   integer(I4P), intent(in)  :: nblk0(3)   !< Blocks per axis at level 0.
   real(R8P),    intent(in)  :: dx0(3)     !< Cell spacing at level 0.
   logical,      intent(in)  :: amr        !< Refine block `refine` to level 1.
   integer(I4P), intent(in)  :: refine(3)  !< The refined level-0 block (0-based).
   integer(I4P), intent(in)  :: root_cells !< Most cells of the multigrid's root (0: ADAM's default).
   logical,      intent(in)  :: periodic   !< Periodic (else ghost-Dirichlet, ADAM's default kinds, zero values).
   logical,      intent(in)  :: exact      !< One V-cycle is the exact solve (uniform, default root).
   integer(I4P), parameter   :: NIJK(3)=[8_I4P, 8_I4P, 8_I4P] !< Cells per block per axis.
   type(grid_object)         :: grid       !< ADAM's grid (shim).
   type(field_object)        :: field      !< ADAM's field (shim).
   type(poisson_mg_object)   :: host       !< ADAM's multigrid solver, host.
   type(MG_DEV_T)            :: dev        !< ADAM's multigrid solver, device.
   integer(I4P), allocatable :: leaves(:,:), coords(:,:) !< Every leaf, this rank's leaves [4,*].
   real(R8P),    allocatable :: bmin(:,:), bmax(:,:)     !< This rank's block corners [3,nb].
   real(R8P),    allocatable :: q(:,:,:,:,:), phi_h(:,:,:,:), phi_d(:,:,:,:) !< Source, potentials.
   real(R8P),    allocatable :: u(:), au(:), x(:), y(:), ax_h(:), ax_d(:)   !< Packed vectors.
   integer(I4P)              :: nleaf, nb, g, b, i, j, k, c, o, lmax, same !< Counters.
   real(R8P)                 :: dx(3), xc(3), vol, s(2), mean, res(2), sym, cmp(4), bound !< Measures.
   logical                   :: is_ref     !< The level-0 block is the refined one.
   integer                   :: ierr_      !< MPI error status.

   ! every leaf, level-0 blocks i fastest, the refined one replaced by its 8 children; filled in place
   nleaf = product(nblk0) + merge(7_I4P, 0_I4P, amr)
   allocate(leaves(4,nleaf))
   g = 0
   do k=0, nblk0(3)-1
      do j=0, nblk0(2)-1
         do i=0, nblk0(1)-1
            is_ref = amr .and. all([i, j, k] == refine)
            if (is_ref) then
               do o=0, 7
                  g = g + 1
                  leaves(:,g) = [2*i + ibits(o, 0, 1), 2*j + ibits(o, 1, 1), 2*k + ibits(o, 2, 1), 1_I4P]
               enddo
            else
               g = g + 1
               leaves(:,g) = [i, j, k, 0_I4P]
            endif
         enddo
      enddo
   enddo
   lmax = merge(1_I4P, 0_I4P, amr)
   call grid%set(nijk=NIJK, ngc=NGHOST, nblk0=nblk0, dx0=dx0, emin=[0._R8P, 0._R8P, 0._R8P], &
                 periodic=[periodic, periodic, periodic], null=[.false., .false., .false.], maxlevel=lmax)
   call deal_blocks(nleaf, leaves, coords)
   nb = size(coords, dim=2)
   allocate(bmin(3,nb), bmax(3,nb))
   do b=1, nb
      bmin(:,b) = grid%domain_emin + coords(1:3,b) * grid%block_dxyz(:,coords(4,b))
      bmax(:,b) = bmin(:,b) + grid%block_dxyz(:,coords(4,b))
   enddo
   call field%set(coords, emin=bmin, emax=bmax)
   allocate(q(1:1,1-NGHOST:NIJK(1)+NGHOST,1-NGHOST:NIJK(2)+NGHOST,1-NGHOST:NIJK(3)+NGHOST,1:max(nb,1)))
   allocate(phi_h(1-NGHOST:NIJK(1)+NGHOST,1-NGHOST:NIJK(2)+NGHOST,1-NGHOST:NIJK(3)+NGHOST,1:max(nb,1)))
   allocate(phi_d, mold=phi_h)
   q = 0._R8P ; phi_h = 0._R8P ; phi_d = 0._R8P
   ! the source at the cell centres; the volume-weighted mean as C0 plus the mean deviation from it
   s = 0._R8P
   do b=1, nb
      dx = grid%cell_dxyz(:,coords(4,b))
      vol = product(dx)
      do k=1, NIJK(3)
         do j=1, NIJK(2)
            do i=1, NIJK(1)
               xc = bmin(:,b) + ([i, j, k] - 0.5_R8P) * dx
               q(1,i,j,k,b) = mg_source(xc, grid%domain_emax - grid%domain_emin)
               s = s + [vol * (q(1,i,j,k,b) - C0), vol]
            enddo
         enddo
      enddo
   enddo
   call MPI_ALLREDUCE(MPI_IN_PLACE, s, 2, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr_)
   mean = C0 + s(1) / s(2)
   if (.not.periodic) mean = 0._R8P ! ghost-Dirichlet, zero values: A phi = factor V f, no mean removed
   if (root_cells > 0) then
      call host%initialize(grid=grid, field=field, tolerance=TOL_MG, max_iterations=MAX_IT, root_cells=root_cells)
      call dev%initialize(grid=grid, field=field, tolerance=TOL_MG, max_iterations=MAX_IT, root_cells=root_cells)
   else
      call host%initialize(grid=grid, field=field, tolerance=TOL_MG, max_iterations=MAX_IT)
      call dev%initialize(grid=grid, field=field, tolerance=TOL_MG, max_iterations=MAX_IT)
   endif
   if (host%lbot /= dev%lbot .or. host%lmax /= dev%lmax) call fail(label//': host and device hierarchies differ')
   call host%solve(field=field, q=q, iv=1_I4P, factor=FACTOR, phi=phi_h)
   call dev%solve(field=field, q=q, iv=1_I4P, factor=FACTOR, phi=phi_d)
   ! the device residual recomputed through its own operator: res(1) max |(A u) / V - factor (f - mean)|, res(2) scale
   c = max(nb*product(NIJK), 1)
   allocate(u(c), au(c), x(c), y(c), ax_h(c), ax_d(c))
   u = 0._R8P ; au = 0._R8P ; ax_h = 0._R8P ; ax_d = 0._R8P
   c = 0
   do b=1, nb
      do k=1, NIJK(3)
         do j=1, NIJK(2)
            do i=1, NIJK(1)
               c = c + 1
               u(c) = phi_d(i,j,k,b)
            enddo
         enddo
      enddo
   enddo
   call dev%apply_a(u, au)
   res = 0._R8P
   c = 0
   do b=1, nb
      vol = product(grid%cell_dxyz(:,coords(4,b)))
      do k=1, NIJK(3)
         do j=1, NIJK(2)
            do i=1, NIJK(1)
               c = c + 1
               res(1) = max(res(1), abs(au(c) / vol - FACTOR * (q(1,i,j,k,b) - mean)))
               res(2) = max(res(2), abs(FACTOR * (q(1,i,j,k,b) - mean)))
            enddo
         enddo
      enddo
   enddo
   call MPI_ALLREDUCE(MPI_IN_PLACE, res, 2, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, ierr_)
   do i=1, size(x)
      x(i) = sin(1.3_R8P * i + 0.7_R8P * mpih%myrank) ; y(i) = cos(2.1_R8P * i - 0.4_R8P * mpih%myrank)
   enddo
   sym = dev%symmetry_defect(x, y)
   ! device against host: the operator on one fixed vector, the potentials
   call host%apply_a(x, ax_h)
   call dev%apply_a(x, ax_d)
   cmp = 0._R8P ; same = 0
   do c=1, nb*product(NIJK)
      cmp(1) = max(cmp(1), abs(ax_d(c) - ax_h(c))) ; cmp(2) = max(cmp(2), abs(ax_h(c)))
   enddo
   do b=1, nb
      do k=1, NIJK(3)
         do j=1, NIJK(2)
            do i=1, NIJK(1)
               cmp(3) = max(cmp(3), abs(phi_d(i,j,k,b) - phi_h(i,j,k,b)))
               cmp(4) = max(cmp(4), abs(phi_h(i,j,k,b)))
               if (transfer(phi_d(i,j,k,b), 0_I8P) == transfer(phi_h(i,j,k,b), 0_I8P)) same = same + 1
            enddo
         enddo
      enddo
   enddo
   call MPI_ALLREDUCE(MPI_IN_PLACE, cmp, 4, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, ierr_)
   call MPI_ALLREDUCE(MPI_IN_PLACE, same, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr_)
   worst(2) = max(worst(2), cmp(1) / cmp(2))
   if (exact) then
      worst(3) = max(worst(3), cmp(3) / cmp(4)) ; bound = TOL_DEV_EXACT
   else
      worst(4) = max(worst(4), cmp(3) / cmp(4)) ; bound = TOL_DEV_FCG_FACTOR * max(dev%tolerance_used, &
                                                                                    host%tolerance_used)
   endif
   if (mpih%myrank == 0) then
      print '(A,I0,A,I0,A,I0,A,I0,A,I0,A,ES9.2,A,ES9.2,A,ES9.2,A,ES9.2)', &
         'elisha_test_adam_device_oracle_mpi: '//label//' leaves ', nleaf, ' levels ', dev%lbot, '..', dev%lmax, &
         ' iterations dev ', dev%iterations, ' host ', host%iterations, ' residual dev ', dev%residual, ' tol ', &
         dev%tolerance_used, ' recomputed ', res(1) / res(2), ' symmetry ', sym
      print '(A,ES9.2,A,ES9.2,A,ES9.2,A,I0,A,I0)', 'elisha_test_adam_device_oracle_mpi: '//label// &
         ' | max rel |A_dev x - A_host x| ', cmp(1) / cmp(2), ' max rel |phi_dev - phi_host| ', cmp(3) / cmp(4), &
         ' (bound ', bound, ') bitwise-equal ', same, ' of ', nleaf * product(NIJK)
      if (dev%iterations /= host%iterations) print '(A)', 'elisha_test_adam_device_oracle_mpi: '//label// &
         ' note: the iteration counts differ (the potentials then differ at the tolerance, not at round-off)'
   endif
   if (.not.(dev%residual <= dev%tolerance_used)) call fail(label//': device FCG residual above its stop tolerance')
   if (.not.(dev%iterations > 0)) call fail(label//': no device FCG iteration')
   if (.not.(res(1) <= 2._R8P * dev%tolerance_used * res(2))) call fail(label//': device recomputed residual above 2 tol')
   if (.not.(sym <= TOL_SYMMETRY)) call fail(label//': device symmetry defect above bound')
   if (.not.(cmp(1) <= TOL_DEV_OP * cmp(2))) call fail(label//': device operator differs from the host one')
   if (.not.(cmp(3) <= bound * cmp(4))) call fail(label//': device potential differs from the host one')
   if (exact .and. (dev%iterations /= 1 .or. host%iterations /= 1)) &
      call fail(label//': the exact V-cycle took more than one FCG iteration')
   call dev%destroy
   call host%destroy
   endsubroutine run_mg

   pure function fft_source(gi, gj, gk, ncell) result(f)
   !< The FFT cases' source at global cell (gi, gj, gk) (0-based) of ncell cells: a constant plus discrete Fourier
   !< modes, one at the Nyquist index along x (`elisha_test_adam_fft_oracle_mpi`'s).
   integer(I4P), intent(in) :: gi, gj, gk !< Global cell.
   integer(I4P), intent(in) :: ncell(3)   !< Global cells per axis.
   real(R8P)                :: f          !< Source.
   real(R8P)                :: t(3)       !< Phases per unit wavenumber.

   t = 2._R8P * PI * [real(gi, R8P) / ncell(1), real(gj, R8P) / ncell(2), real(gk, R8P) / ncell(3)]
   f = C0 + cos(t(1)) + 0.5_R8P * cos(2._R8P * t(2) + t(3) + 0.3_R8P) &
       + 0.25_R8P * sin(3._R8P * t(1) + t(2) + 2._R8P * t(3)) + 0.125_R8P * cos((ncell(1) / 2) * t(1) + t(2))
   endfunction fft_source

   pure function mg_source(xc, length) result(f)
   !< The multigrid cases' source at the point xc of a box of sides `length` (lower corner at the origin): a constant
   !< plus smooth periodic modes (`elisha_test_adam_mg_oracle_mpi`'s).
   real(R8P), intent(in) :: xc(3)     !< Point.
   real(R8P), intent(in) :: length(3) !< Box sides.
   real(R8P)             :: f         !< Source.
   real(R8P)             :: t(3)      !< Phases per unit wavenumber.

   t = 2._R8P * PI * xc / length
   f = C0 + cos(t(1)) + 0.5_R8P * cos(2._R8P * t(2) + t(3) + 0.3_R8P) &
       + 0.25_R8P * sin(3._R8P * t(1) + t(2) + 2._R8P * t(3))
   endfunction mg_source

   subroutine fail(msg)
   !< Report a failed check.
   character(*), intent(in) :: msg !< Message.

   failures = failures + 1
   print '(A,I0,A)', 'FAIL (rank ', mpih%myrank, '): '//msg
   endsubroutine fail
endprogram elisha_test_adam_device_oracle_mpi
