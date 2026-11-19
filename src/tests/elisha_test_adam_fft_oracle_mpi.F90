!< ELISHA test: ADAM's FFT Poisson solver runs inside ELISHA's tests as an oracle, and solves what it claims.
program elisha_test_adam_fft_oracle_mpi
!< ELISHA test: ADAM's FFT Poisson solver runs inside ELISHA's tests as an oracle, and solves what it claims.
!<
!< The solver is `adam_poisson_fft_object`, ADAM 669d0bdd verbatim, over the test shims `adam_grid_object`,
!< `adam_field_object`, `adam_mpih_global`. On uniform periodic grids (cubic 3-D at level 0, non-cubic cells and domain
!< at level 1, 2-D with a null z axis), the blocks dealt round-robin to the ranks, the source a constant plus a few
!< discrete Fourier modes (one at the Nyquist index): by meaning, the 7-point (5-point in 2-D) Laplacian of the returned
!< `phi`, read through the ghost layers the solver fills, equals `factor (f - mean f)` to a relative max-norm of 1e-12.
!< ADAM's solver needs MPI: built with `-DELISHA_NO_MPI` (the serial modes, fpm) the program prints a skip line and
!< succeeds. The FP traps of a debug build are off around `MPI_Init` only (hwloc raises IEEE flags inside it).
#ifndef ELISHA_NO_MPI
use :: mpi
use, intrinsic :: ieee_exceptions, only : ieee_usual, ieee_get_halting_mode, ieee_set_halting_mode
use :: penf, only : I4P, R8P
use :: adam_grid_object, only : grid_object
use :: adam_field_object, only : field_object
use :: adam_mpih_global, only : mpih
use :: adam_poisson_fft_object, only : poisson_fft_object
implicit none
real(R8P),    parameter :: TOL=1.e-12_R8P  !< Relative max-norm residual allowed.
real(R8P),    parameter :: FACTOR=2.5_R8P  !< Source factor.
real(R8P),    parameter :: PI=acos(-1._R8P) !< Pi.
real(R8P),    parameter :: C0=0.7_R8P      !< The constant of the source (its mean: the modes sum to zero).
integer(I4P)            :: failures, nfail !< Failed checks, on this rank and over the ranks.
integer                 :: ierr            !< MPI error status.
logical                 :: halting(3)      !< The FP halting modes of the build (debug: on).

failures = 0
! MPI_Init runs C code (hwloc's XML backend) that raises IEEE flags: a debug build's FP traps (-ffpe-trap) would
! kill it there, so they are off around the call only (Open MPI 5.0.11, hwloc 2.7.1)
call ieee_get_halting_mode(ieee_usual, halting)
call ieee_set_halting_mode(ieee_usual, .false.)
call MPI_INIT(ierr)
call ieee_set_halting_mode(ieee_usual, halting)
call mpih%initialize
call run_case(label='cubic, level 0      ', nijk=[8_I4P, 8_I4P, 8_I4P], nblk0=[2_I4P, 2_I4P, 2_I4P], level=0_I4P, &
              dx0=[1._R8P, 1._R8P, 1._R8P] / 16._R8P, emin=[0._R8P, 0._R8P, 0._R8P], null=[.false., .false., .false.])
call run_case(label='non-cubic, level 1  ', nijk=[8_I4P, 4_I4P, 6_I4P], nblk0=[1_I4P, 2_I4P, 1_I4P], level=1_I4P, &
              dx0=[0.1_R8P, 0.07_R8P, 0.13_R8P], emin=[-0.3_R8P, 0.2_R8P, 1._R8P], null=[.false., .false., .false.])
call run_case(label='2-D (null z), lvl 0 ', nijk=[8_I4P, 8_I4P, 1_I4P], nblk0=[2_I4P, 3_I4P, 1_I4P], level=0_I4P, &
              dx0=[0.05_R8P, 0.08_R8P, 1._R8P], emin=[0._R8P, 0._R8P, 0._R8P], null=[.false., .false., .true.])
nfail = failures
call MPI_ALLREDUCE(MPI_IN_PLACE, nfail, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)
if (mpih%myrank == 0 .and. nfail == 0) print '(A,I0,A)', 'elisha_test_adam_fft_oracle_mpi: ok (', mpih%procs_number, &
                                                        ' rank(s))'
call MPI_FINALIZE(ierr)
if (nfail > 0) error stop 1

contains
   subroutine run_case(label, nijk, nblk0, level, dx0, emin, null)
   !< One grid: blocks dealt round-robin, ADAM's solve, the Laplacian of phi against factor (f - mean f).
   character(*), intent(in)  :: label    !< Case label.
   integer(I4P), intent(in)  :: nijk(3)  !< Cells per block per axis.
   integer(I4P), intent(in)  :: nblk0(3) !< Blocks per axis at level 0.
   integer(I4P), intent(in)  :: level    !< Level of every block.
   real(R8P),    intent(in)  :: dx0(3)   !< Cell spacing at level 0.
   real(R8P),    intent(in)  :: emin(3)  !< Domain lower corner.
   logical,      intent(in)  :: null(3)  !< Null axes.
   integer(I4P), parameter   :: NGHOST=2_I4P !< Ghost layers.
   type(grid_object)         :: grid     !< ADAM's grid (shim).
   type(field_object)        :: field    !< ADAM's field (shim).
   type(poisson_fft_object)  :: solver   !< ADAM's FFT solver.
   integer(I4P), allocatable :: coords(:,:) !< This rank's blocks [4,nb].
   real(R8P),    allocatable :: q(:,:,:,:,:), phi(:,:,:,:) !< Source (one variable), potential.
   integer(I4P)              :: nbl(3), ncell(3), nb, b, g, i, j, k, gi, gj, gk !< Blocks, cells, counters.
   real(R8P)                 :: dx(3), mean, lap, res(2) !< Spacing, source mean, Laplacian, residual and scale.
   integer                   :: ierr_    !< MPI error status.

   call grid%set(nijk=nijk, ngc=NGHOST, nblk0=nblk0, dx0=dx0, emin=emin, periodic=.not.null, null=null, maxlevel=level)
   nbl = merge(nblk0, nblk0 * 2_I4P**level, null)
   ncell = nbl * nijk
   dx = grid%cell_dxyz(:,level)
   ! deal the blocks round-robin: global block g (i fastest, 0-based) to rank mod(g, ranks); fill in place
   nb = 0
   do g=0, product(nbl)-1
      if (mod(g, mpih%procs_number) == mpih%myrank) nb = nb + 1
   enddo
   allocate(coords(4,nb))
   b = 0
   do g=0, product(nbl)-1
      if (mod(g, mpih%procs_number) /= mpih%myrank) cycle
      b = b + 1
      coords(:,b) = [mod(g, nbl(1)), mod(g / nbl(1), nbl(2)), g / (nbl(1) * nbl(2)), level]
   enddo
   call field%set(coords)
   allocate(q(1:1,1-NGHOST:nijk(1)+NGHOST,1-NGHOST:nijk(2)+NGHOST,1-NGHOST:nijk(3)+NGHOST,1:max(nb,1)))
   allocate(phi(1-NGHOST:nijk(1)+NGHOST,1-NGHOST:nijk(2)+NGHOST,1-NGHOST:nijk(3)+NGHOST,1:max(nb,1)))
   q = 0._R8P ; phi = 0._R8P
   ! the mean as C0 plus the mean deviation from it: summing the O(1) deviations, not the values, keeps the sum's
   ! round-off (rank-order dependent) below the residual measured
   mean = 0._R8P
   do b=1, nb
      do k=1, nijk(3)
         do j=1, nijk(2)
            do i=1, nijk(1)
               gi = coords(1,b) * nijk(1) + i - 1 ; gj = coords(2,b) * nijk(2) + j - 1
               gk = coords(3,b) * nijk(3) + k - 1
               q(1,i,j,k,b) = source(gi, gj, gk, ncell)
               mean = mean + (q(1,i,j,k,b) - C0)
            enddo
         enddo
      enddo
   enddo
   call MPI_ALLREDUCE(MPI_IN_PLACE, mean, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr_)
   mean = C0 + mean / real(product(ncell), R8P)
   call solver%initialize(grid=grid, field=field)
   call solver%solve(field=field, q=q, iv=1_I4P, factor=FACTOR, phi=phi)
   ! res(1): max |L phi - factor (f - mean)|, res(2): max |factor (f - mean)|
   res = 0._R8P
   do b=1, nb
      do k=1, nijk(3)
         do j=1, nijk(2)
            do i=1, nijk(1)
               lap = (phi(i+1,j,k,b) - 2._R8P * phi(i,j,k,b) + phi(i-1,j,k,b)) / dx(1)**2 + &
                     (phi(i,j+1,k,b) - 2._R8P * phi(i,j,k,b) + phi(i,j-1,k,b)) / dx(2)**2
               if (.not.null(3)) lap = lap + (phi(i,j,k+1,b) - 2._R8P * phi(i,j,k,b) + phi(i,j,k-1,b)) / dx(3)**2
               res(1) = max(res(1), abs(lap - FACTOR * (q(1,i,j,k,b) - mean)))
               res(2) = max(res(2), abs(FACTOR * (q(1,i,j,k,b) - mean)))
            enddo
         enddo
      enddo
   enddo
   call MPI_ALLREDUCE(MPI_IN_PLACE, res, 2, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, ierr_)
   if (mpih%myrank == 0) print '(A,I0,A,3(I0,1X),A,ES10.3)', 'elisha_test_adam_fft_oracle_mpi: '//label//' np ', &
      mpih%procs_number, ' cells ', ncell, 'residual ', res(1) / res(2)
   if (.not.(res(1) <= TOL * res(2))) call fail(label//': residual above 1e-12')
   call solver%destroy
   endsubroutine run_case

   pure function source(gi, gj, gk, ncell) result(f)
   !< The source at global cell (gi, gj, gk) (0-based) of ncell cells: a constant plus discrete Fourier modes, one at
   !< the Nyquist index along x.
   integer(I4P), intent(in) :: gi, gj, gk !< Global cell.
   integer(I4P), intent(in) :: ncell(3)   !< Global cells per axis.
   real(R8P)                :: f          !< Source.
   real(R8P)                :: t(3)       !< Phases per unit wavenumber.

   t = 2._R8P * PI * [real(gi, R8P) / ncell(1), real(gj, R8P) / ncell(2), real(gk, R8P) / ncell(3)]
   f = C0 + cos(t(1)) + 0.5_R8P * cos(2._R8P * t(2) + t(3) + 0.3_R8P) &
       + 0.25_R8P * sin(3._R8P * t(1) + t(2) + 2._R8P * t(3)) + 0.125_R8P * cos((ncell(1) / 2) * t(1) + t(2))
   endfunction source

   subroutine fail(msg)
   !< Report a failed check.
   character(*), intent(in) :: msg !< Message.

   failures = failures + 1
   print '(A)', 'FAIL: '//msg
   endsubroutine fail
#else
implicit none
print '(A)', 'elisha_test_adam_fft_oracle_mpi: skipped (ADAM''s FFT solver needs MPI; built with -DELISHA_NO_MPI)'
#endif
endprogram elisha_test_adam_fft_oracle_mpi
