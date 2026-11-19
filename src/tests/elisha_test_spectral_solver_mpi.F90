!< ELISHA test: the spectral solver solves what it claims on any number of ranks (and without MPI), and refuses.
program elisha_test_spectral_solver_mpi
!< ELISHA test: the spectral solver solves what it claims on any number of ranks (and without MPI), and refuses.
!<
!< Smoke test by meaning (the parity with ADAM's solver is another test's). On uniform periodic meshes (cubic 3-D at
!< level 0, non-cubic cells and domain at level 1, 2-D with a null z axis), the blocks dealt round-robin to the ranks,
!< the source a constant plus a few discrete Fourier modes (one at the Nyquist index): through `setup` and
!< `solve_packed`, the 7-point (5-point in 2-D) Laplacian of `phi`, read through the ghost layers the solver fills,
!< equals `factor (f - mean f)` to a relative max-norm of 1e-12. Refusals: a mesh at two levels
!< (`ELISHA_ERR_MESH_LEVELS`), a solve before the setup (`ELISHA_ERR_NOT_INITIALIZED`), a `phi` of the wrong shape
!< (`ELISHA_ERR_SHAPE`), each with the same status on every rank. The FP traps of a debug build are off around
!< `MPI_Init` only (hwloc raises IEEE flags inside it).
#ifndef ELISHA_NO_MPI
use mpi
#endif
use, intrinsic :: ieee_exceptions, only : ieee_usual, ieee_get_halting_mode, ieee_set_halting_mode
use elisha, only : I4P, R_P, ELISHA_OK, ELISHA_ERR_MESH_LEVELS, ELISHA_ERR_NOT_INITIALIZED, ELISHA_ERR_SHAPE, &
                   elisha_comm_object, elisha_mesh_object, elisha_spectral_solver_object
implicit none
real(R_P), parameter     :: TOL=1.e-12_R_P   !< Relative max-norm residual allowed.
real(R_P), parameter     :: FACTOR=2.5_R_P   !< Source factor.
real(R_P), parameter     :: PI=acos(-1._R_P) !< Pi.
real(R_P), parameter     :: C0=0.7_R_P       !< The constant of the source (its mean: the modes sum to zero).
type(elisha_comm_object) :: c                !< Communication context.
integer(I4P)             :: failures, nfail(1), stat !< Failed checks, over the ranks; status.
integer                  :: wrank, wsize, world, ierr !< MPI.
logical                  :: halting(3)       !< The FP halting modes of the build (debug: on).
character(:), allocatable :: msg             !< Message.

failures = 0
#ifndef ELISHA_NO_MPI
call ieee_get_halting_mode(ieee_usual, halting)
call ieee_set_halting_mode(ieee_usual, .false.)
call MPI_INIT(ierr)
call ieee_set_halting_mode(ieee_usual, halting)
world = MPI_COMM_WORLD
#else
halting = .false. ; ierr = 0 ; world = 0
#endif
call c%init(world, stat, msg) ; if (stat /= ELISHA_OK) call fail('comm init: '//msg)
wrank = c%rank ; wsize = c%nranks
call run_case(label='cubic, level 0      ', nijk=[8_I4P, 8_I4P, 8_I4P], nblk0=[2_I4P, 2_I4P, 2_I4P], level=0_I4P, &
              dx0=[1._R_P, 1._R_P, 1._R_P] / 16._R_P, emin=[0._R_P, 0._R_P, 0._R_P], null=[.false., .false., .false.])
call run_case(label='non-cubic, level 1  ', nijk=[8_I4P, 4_I4P, 6_I4P], nblk0=[1_I4P, 2_I4P, 1_I4P], level=1_I4P, &
              dx0=[0.1_R_P, 0.07_R_P, 0.13_R_P], emin=[-0.3_R_P, 0.2_R_P, 1._R_P], null=[.false., .false., .false.])
call run_case(label='2-D (null z), lvl 0 ', nijk=[8_I4P, 8_I4P, 1_I4P], nblk0=[2_I4P, 3_I4P, 1_I4P], level=0_I4P, &
              dx0=[0.05_R_P, 0.08_R_P, 1._R_P], emin=[0._R_P, 0._R_P, 0._R_P], null=[.false., .false., .true.])
call check_refusals
nfail = failures
call c%max_i4(nfail, stat)
call c%destroy(stat)
#ifndef ELISHA_NO_MPI
call MPI_FINALIZE(ierr)
#endif
if (nfail(1) > 0) error stop 1
if (wrank == 0) print '(A,I0,A)', 'elisha_test_spectral_solver_mpi: ok (', wsize, ' rank(s))'

contains
   subroutine run_case(label, nijk, nblk0, level, dx0, emin, null)
   !< One mesh: blocks dealt round-robin, the solve, the Laplacian of phi against factor (f - mean f).
   character(*), intent(in)  :: label    !< Case label.
   integer(I4P), intent(in)  :: nijk(3)  !< Cells per block per axis.
   integer(I4P), intent(in)  :: nblk0(3) !< Blocks per axis at level 0.
   integer(I4P), intent(in)  :: level    !< Level of every block.
   real(R_P),    intent(in)  :: dx0(3)   !< Cell spacing at level 0.
   real(R_P),    intent(in)  :: emin(3)  !< Domain lower corner.
   logical,      intent(in)  :: null(3)  !< Null axes.
   integer(I4P), parameter   :: NGHOST=2_I4P !< Ghost layers.
   type(elisha_mesh_object)            :: mesh   !< Mesh.
   type(elisha_spectral_solver_object) :: solver !< Solver.
   integer(I4P), allocatable :: coords(:,:) !< This rank's blocks [4,nb].
   real(R_P),    allocatable :: buf(:), phi(:,:,:,:) !< Packed source, potential.
   real(R_P),    allocatable :: f(:,:,:,:) !< Source.
   integer(I4P)              :: nbl(3), ncell(3), nb, b, g, i, j, k, gi, gj, gk, n !< Blocks, cells, counters.
   real(R_P)                 :: dx(3), mean(1), lap, res(2) !< Spacing, source mean, Laplacian, residual and scale.

   nbl = merge(nblk0, nblk0 * 2_I4P**level, null)
   ncell = nbl * nijk
   nb = 0
   do g=0, product(nbl)-1
      if (mod(g, wsize) == wrank) nb = nb + 1
   enddo
   allocate(coords(4,nb))
   b = 0
   do g=0, product(nbl)-1
      if (mod(g, wsize) /= wrank) cycle
      b = b + 1
      coords(:,b) = [mod(g, nbl(1)), mod(g / nbl(1), nbl(2)), g / (nbl(1) * nbl(2)), level]
   enddo
   call mesh%init(nijk, NGHOST, nblk0, dx0, emin, .not.null, null, coords, stat, msg)
   if (stat /= ELISHA_OK) then
      call fail(label//': mesh init: '//msg)
      return
   endif
   call solver%setup(mesh, c, stat, msg)
   if (stat /= ELISHA_OK) then
      call fail(label//': setup: '//msg)
      return
   endif
   dx = mesh%cell_dx(level)
   allocate(f(nijk(1),nijk(2),nijk(3),max(nb,1)), buf(max(nb*product(nijk),1)))
   allocate(phi(1-NGHOST:nijk(1)+NGHOST,1-NGHOST:nijk(2)+NGHOST,1-NGHOST:nijk(3)+NGHOST,1:max(nb,1)))
   phi = 0._R_P
   ! the mean as C0 plus the mean deviation from it: summing the O(1) deviations keeps the sum's round-off low
   mean = 0._R_P ; n = 0
   do b=1, nb
      do k=1, nijk(3)
         do j=1, nijk(2)
            do i=1, nijk(1)
               gi = coords(1,b) * nijk(1) + i - 1 ; gj = coords(2,b) * nijk(2) + j - 1
               gk = coords(3,b) * nijk(3) + k - 1
               f(i,j,k,b) = source(gi, gj, gk, ncell)
               n = n + 1 ; buf(n) = f(i,j,k,b)
               mean(1) = mean(1) + (f(i,j,k,b) - C0)
            enddo
         enddo
      enddo
   enddo
   call c%sum_r(mean, stat)
   mean(1) = C0 + mean(1) / real(product(ncell), R_P)
   call solver%solve_packed(mesh, buf, FACTOR, phi, stat, msg)
   if (stat /= ELISHA_OK) then
      call fail(label//': solve: '//msg)
      return
   endif
   ! res(1): max |L phi - factor (f - mean)|, res(2): max |factor (f - mean)|
   res = 0._R_P
   do b=1, nb
      do k=1, nijk(3)
         do j=1, nijk(2)
            do i=1, nijk(1)
               lap = (phi(i+1,j,k,b) - 2._R_P * phi(i,j,k,b) + phi(i-1,j,k,b)) / dx(1)**2 + &
                     (phi(i,j+1,k,b) - 2._R_P * phi(i,j,k,b) + phi(i,j-1,k,b)) / dx(2)**2
               if (.not.null(3)) lap = lap + (phi(i,j,k+1,b) - 2._R_P * phi(i,j,k,b) + phi(i,j,k-1,b)) / dx(3)**2
               res(1) = max(res(1), abs(lap - FACTOR * (f(i,j,k,b) - mean(1))))
               res(2) = max(res(2), abs(FACTOR * (f(i,j,k,b) - mean(1))))
            enddo
         enddo
      enddo
   enddo
   call allmax(res)
   if (wrank == 0) print '(A,I0,A,3(I0,1X),A,ES10.3)', 'elisha_test_spectral_solver_mpi: '//label//' np ', wsize, &
      ' cells ', ncell, 'residual ', res(1) / res(2)
   if (.not.(res(1) <= TOL * res(2))) call fail(label//': residual above 1e-12')
   call solver%destroy
   call mesh%destroy
   endsubroutine run_case

   subroutine check_refusals
   !< A mesh at two levels, a solve before the setup, a phi of the wrong shape: each code on every rank.
   type(elisha_mesh_object)            :: mesh   !< Mesh.
   type(elisha_spectral_solver_object) :: solver !< Solver.
   integer(I4P), allocatable           :: coords(:,:) !< This rank's blocks.
   real(R_P),    allocatable           :: buf(:), phi(:,:,:,:) !< Source, potential.
   integer(I4P)                        :: g, b, nb  !< Counters.

   ! two levels (rank 0: a level-0 block and a level-1 one): validate refuses the uniform mesh first
   if (wrank == 0) then
      coords = reshape([0_I4P, 0_I4P, 0_I4P, 0_I4P, 1_I4P, 1_I4P, 1_I4P, 1_I4P], [4, 2])
   else
      allocate(coords(4,0))
   endif
   call mesh%init([4_I4P, 4_I4P, 4_I4P], 1_I4P, [1_I4P, 1_I4P, 1_I4P], [0.25_R_P, 0.25_R_P, 0.25_R_P], &
                  [0._R_P, 0._R_P, 0._R_P], [.true., .true., .true.], [.false., .false., .false.], coords, stat, msg)
   if (stat /= ELISHA_OK) call fail('refusal mesh init: '//msg)
   call solver%setup(mesh, c, stat, msg)
   if (stat /= ELISHA_ERR_MESH_LEVELS) call fail('two levels not refused with ELISHA_ERR_MESH_LEVELS')
   call mesh%destroy
   ! a valid mesh: 2 x 2 x 2 blocks of 4^3 cells, round-robin
   nb = 0
   do g=0, 7
      if (mod(g, wsize) == wrank) nb = nb + 1
   enddo
   deallocate(coords) ; allocate(coords(4,nb))
   b = 0
   do g=0, 7
      if (mod(g, wsize) /= wrank) cycle
      b = b + 1 ; coords(:,b) = [mod(g, 2), mod(g / 2, 2), g / 4, 0]
   enddo
   call mesh%init([4_I4P, 4_I4P, 4_I4P], 1_I4P, [2_I4P, 2_I4P, 2_I4P], [0.125_R_P, 0.125_R_P, 0.125_R_P], &
                  [0._R_P, 0._R_P, 0._R_P], [.true., .true., .true.], [.false., .false., .false.], coords, stat, msg)
   if (stat /= ELISHA_OK) call fail('refusal mesh init (valid): '//msg)
   allocate(buf(max(nb*64,1))) ; buf = 1._R_P
   allocate(phi(0:5,0:5,0:5,max(nb,1))) ; phi = 0._R_P
   ! a solve before the setup
   call solver%solve_packed(mesh, buf, FACTOR, phi, stat, msg)
   if (stat /= ELISHA_ERR_NOT_INITIALIZED) call fail('solve before setup not refused with ELISHA_ERR_NOT_INITIALIZED')
   call solver%setup(mesh, c, stat, msg)
   if (stat /= ELISHA_OK) call fail('refusal setup (valid): '//msg)
   ! phi without its upper ghost layer: [0:4] per direction instead of [0:5]
   deallocate(phi) ; allocate(phi(0:4,0:4,0:4,max(nb,1))) ; phi = 0._R_P
   call solver%solve_packed(mesh, buf, FACTOR, phi, stat, msg)
   if (stat /= ELISHA_ERR_SHAPE) call fail('a wrong phi shape not refused with ELISHA_ERR_SHAPE')
   call solver%destroy
   call mesh%destroy
   endsubroutine check_refusals

   pure function source(gi, gj, gk, ncell) result(f)
   !< The source at global cell (gi, gj, gk) (0-based) of ncell cells: a constant plus discrete Fourier modes, one at
   !< the Nyquist index along x.
   integer(I4P), intent(in) :: gi, gj, gk !< Global cell.
   integer(I4P), intent(in) :: ncell(3)   !< Global cells per axis.
   real(R_P)                :: f          !< Source.
   real(R_P)                :: t(3)       !< Phases per unit wavenumber.

   t = 2._R_P * PI * [real(gi, R_P) / ncell(1), real(gj, R_P) / ncell(2), real(gk, R_P) / ncell(3)]
   f = C0 + cos(t(1)) + 0.5_R_P * cos(2._R_P * t(2) + t(3) + 0.3_R_P) &
       + 0.25_R_P * sin(3._R_P * t(1) + t(2) + 2._R_P * t(3)) + 0.125_R_P * cos((ncell(1) / 2) * t(1) + t(2))
   endfunction source

   subroutine allmax(x)
   !< x <- its elementwise maximum over the ranks (the identity without MPI).
   real(R_P), intent(inout) :: x(:) !< Values.
#ifndef ELISHA_NO_MPI
   integer                  :: ierr_ !< MPI error status.

   call MPI_ALLREDUCE(MPI_IN_PLACE, x, size(x), MPI_DOUBLE_PRECISION, MPI_MAX, c%comm, ierr_)
#else
   if (.false.) x = -x
#endif
   endsubroutine allmax

   subroutine fail(what)
   !< Report a failed check.
   character(*), intent(in) :: what !< What failed.

   failures = failures + 1
   print '(A,I0,A)', '  FAIL rank ', wrank, ': '//what
   endsubroutine fail
endprogram elisha_test_spectral_solver_mpi
