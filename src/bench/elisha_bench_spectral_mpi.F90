!< ELISHA benchmark: the spectral solver's solve time against ADAM's (the verbatim oracle) on the same inputs.
program elisha_bench_spectral_mpi
!< ELISHA benchmark: the spectral solver's solve time against ADAM's (the verbatim oracle) on the same inputs.
!<
!< The cube [-1,1]^3 with a Gaussian source (ADAM's FLAIL "Performance" problem), uniform: 128^3 cells (4^3 blocks of
!< 32^3) periodic and isolated, and 64^3 cells (4^3 blocks of 16^3) periodic. The blocks are dealt round-robin. Both
!< solvers get the same packed source; their potentials must be bitwise equal (checked once per case). Timing: wall
!< clock (`MPI_Wtime`), a barrier before every solve, ADAM and ELISHA solves alternated (A, E, A, E, ...) after one
!< untimed pair, each solve's time the maximum over the ranks, the minimum over the repetitions reported (the median
!< too); the setups are timed once each, apart. Output: one row per case. Not a test (`scripts/run_tests.sh` runs
!< `exe/` only; fpm does not build `src/bench`). Usage: `mpirun -np N exe-bench/elisha_bench_spectral_mpi [reps [EA]]`:
!< `EA` times ELISHA first in every pair (a control for an order bias), the default is ADAM first.
use mpi
use, intrinsic :: ieee_exceptions, only : ieee_usual, ieee_get_halting_mode, ieee_set_halting_mode
use, intrinsic :: iso_fortran_env, only : int64
use elisha, only : I4P, R_P, ELISHA_OK, elisha_comm_object, elisha_mesh_object, elisha_spectral_solver_object
use adam_grid_object, only : grid_object
use adam_field_object, only : field_object
use adam_mpih_global, only : mpih
use adam_poisson_fft_object, only : poisson_fft_object
implicit none
real(R_P), parameter     :: FACTOR=4._R_P * acos(-1._R_P) !< Source factor (4 pi G, G = 1).
type(elisha_comm_object) :: c          !< Communication context.
integer(I4P)             :: stat, reps !< Status, repetitions.
integer                  :: ierr, na   !< MPI error status, arguments.
logical                  :: halting(3) !< The FP halting modes of the build.
logical                  :: efirst=.false. !< ELISHA first in every pair.
character(:), allocatable :: msg       !< Message.
character(16)            :: arg        !< Argument.

call ieee_get_halting_mode(ieee_usual, halting)
call ieee_set_halting_mode(ieee_usual, .false.)
call MPI_INIT(ierr)
call ieee_set_halting_mode(ieee_usual, halting)
call mpih%initialize
call c%init(MPI_COMM_WORLD, stat, msg)
if (stat /= ELISHA_OK) error stop 'comm init'
reps = 9
na = command_argument_count()
if (na > 0) then
   call get_command_argument(1, arg)
   read(arg, *) reps
endif
if (na > 1) then
   call get_command_argument(2, arg)
   efirst = trim(arg) == 'EA'
endif
if (c%rank == 0) then
   print '(A,I0,A,I0,A,A)', '# elisha_bench_spectral_mpi: ', c%nranks, ' rank(s), ', reps, &
         ' repetitions (min of max over ranks), order ', merge('E, A', 'A, E', efirst)
   print '(A)', '# case                 path  setup_A[s] setup_E[s]  solve_A[s]  solve_E[s]  E/A(min)  med_A[s]   '// &
                'med_E[s]   E/A(med)  bitwise'
endif
call run_case('128^3 periodic', 32, .false.)
call run_case('128^3 isolated', 32, .true.)
call run_case(' 64^3 periodic', 16, .false.)
call c%destroy(stat)
call MPI_FINALIZE(ierr)

contains
   subroutine run_case(label, nc, isolated)
   !< One case: 4^3 blocks of nc^3 cells on [-1,1]^3, periodic or isolated; both setups, then the alternated solves.
   character(*), intent(in) :: label    !< Case label.
   integer,      intent(in) :: nc       !< Cells per block per axis.
   logical,      intent(in) :: isolated !< Isolated boundaries.
   integer(I4P), parameter  :: NGC=2_I4P, NB0=4_I4P !< Ghost layers, blocks per axis.
   real(R_P),    parameter  :: SIGMA=0.1_R_P        !< Gaussian width.
   type(grid_object)                   :: grid  !< ADAM's grid (shim).
   type(field_object)                  :: field !< ADAM's field (shim).
   type(poisson_fft_object)            :: adam  !< ADAM's solver.
   type(elisha_mesh_object)            :: mesh  !< ELISHA's mesh.
   type(elisha_spectral_solver_object) :: eli   !< ELISHA's solver.
   integer(I4P), allocatable :: c4(:,:)          !< This rank's blocks.
   real(R_P),    allocatable :: buf(:), pa(:,:,:,:), pe(:,:,:,:) !< Source, potentials.
   real(R_P),    allocatable :: ta(:), te(:)     !< Solve times per repetition.
   real(R_P)                 :: dx, x(3), sa, se, t0 !< Spacing, cell centre, setup times, mark.
   integer(I4P)              :: n, b, g, i, j, k, m, r, slot, nbad(1) !< Counters, mismatches.
   character(4)              :: path         !< Path.

   dx = 2._R_P / (NB0 * nc)
   n = 0
   do g=0, NB0**3-1
      if (mod(g, c%nranks) == c%rank) n = n + 1
   enddo
   allocate(c4(4,n))
   b = 0
   do g=0, NB0**3-1
      if (mod(g, c%nranks) /= c%rank) cycle
      b = b + 1 ; c4(:,b) = [mod(g, NB0), mod(g / NB0, NB0), g / NB0**2, 0_I4P]
   enddo
   call grid%set(nijk=[nc, nc, nc], ngc=NGC, nblk0=[NB0, NB0, NB0], dx0=[dx, dx, dx], emin=[-1._R_P, -1._R_P, -1._R_P], &
                 periodic=spread(.not.isolated, 1, 3), null=[.false., .false., .false.], maxlevel=0_I4P)
   call field%set(c4)
   call mesh%init([nc, nc, nc], NGC, [NB0, NB0, NB0], [dx, dx, dx], [-1._R_P, -1._R_P, -1._R_P], &
                  spread(.not.isolated, 1, 3), [.false., .false., .false.], c4, stat, msg)
   if (stat /= ELISHA_OK) error stop 'mesh init'
   allocate(buf(max(n*nc**3,1)))
   m = 0
   do b=1, n
      do k=1, nc
         do j=1, nc
            do i=1, nc
               x = -1._R_P + ([c4(1,b) * nc + i, c4(2,b) * nc + j, c4(3,b) * nc + k] - 0.5_R_P) * dx
               m = m + 1 ; buf(m) = exp(-sum(x**2) / (2._R_P * SIGMA**2))
            enddo
         enddo
      enddo
   enddo
   allocate(pa(1-NGC:nc+NGC,1-NGC:nc+NGC,1-NGC:nc+NGC,max(n,1)), pe(1-NGC:nc+NGC,1-NGC:nc+NGC,1-NGC:nc+NGC,max(n,1)))
   pa = 0._R_P ; pe = 0._R_P
   ! the setups, once each
   call MPI_BARRIER(MPI_COMM_WORLD, ierr) ; t0 = MPI_WTIME()
   call adam%initialize(grid=grid, field=field, isolated=isolated)
   sa = tmax(MPI_WTIME() - t0)
   call MPI_BARRIER(MPI_COMM_WORLD, ierr) ; t0 = MPI_WTIME()
   call eli%setup(mesh, c, stat, msg, isolated=isolated)
   se = tmax(MPI_WTIME() - t0)
   if (stat /= ELISHA_OK) error stop 'setup'
   ! one untimed pair, then the alternated timed solves
   call adam%solve_packed(field=field, buf=buf, factor=FACTOR, phi=pa)
   call eli%solve_packed(mesh, buf, FACTOR, pe, stat)
   nbad = count(transfer(pa(:,:,:,1:n), [0_int64]) /= transfer(pe(:,:,:,1:n), [0_int64]))
   call MPI_ALLREDUCE(MPI_IN_PLACE, nbad, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)
   allocate(ta(reps), te(reps))
   do r=1, reps
      do slot=1, 2
         call MPI_BARRIER(MPI_COMM_WORLD, ierr) ; t0 = MPI_WTIME()
         if ((slot == 1) .neqv. efirst) then
            call adam%solve_packed(field=field, buf=buf, factor=FACTOR, phi=pa)
            ta(r) = tmax(MPI_WTIME() - t0)
         else
            call eli%solve_packed(mesh, buf, FACTOR, pe, stat)
            te(r) = tmax(MPI_WTIME() - t0)
         endif
      enddo
   enddo
   path = merge('slab', 'rk0 ', eli%slab)
   if (c%rank == 0) print '(A,2X,A4,2F11.4,2F12.5,F10.4,2F11.5,F10.4,2X,A)', label, path, sa, se, minval(ta), &
      minval(te), minval(te) / minval(ta), median(ta), median(te), median(te) / median(ta), &
      merge('yes', 'NO ', nbad(1) == 0)
   call adam%destroy
   call eli%destroy
   call mesh%destroy
   endsubroutine run_case

   function tmax(t) result(tm)
   !< The maximum of t over the ranks.
   real(R_P), intent(in) :: t  !< This rank's time.
   real(R_P)             :: tm !< Maximum.
   integer               :: ierr_ !< MPI error status.

   tm = t
   call MPI_ALLREDUCE(MPI_IN_PLACE, tm, 1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, ierr_)
   endfunction tmax

   pure function median(t) result(md)
   !< The median of t (insertion sort of a copy).
   real(R_P), intent(in) :: t(:) !< Values.
   real(R_P)             :: md   !< Median.
   real(R_P)             :: s(size(t)), v !< Sorted copy, value.
   integer               :: i, j !< Counters.

   s = t
   do i=2, size(s)
      v = s(i) ; j = i - 1
      do while (j >= 1)
         if (s(j) <= v) exit
         s(j+1) = s(j) ; j = j - 1
      enddo
      s(j+1) = v
   enddo
   if (mod(size(s), 2) == 1) then
      md = s((size(s) + 1) / 2)
   else
      md = 0.5_R_P * (s(size(s) / 2) + s(size(s) / 2 + 1))
   endif
   endfunction median
endprogram elisha_bench_spectral_mpi
