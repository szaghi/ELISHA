!< Quick start: a uniform periodic mesh, one solve, the residual through the ghost layers and the analytic error.
program quickstart
!run quickstart quickstart
!region use
use elisha
implicit none
!endregion use
integer(I4P), parameter :: NB=2, NC=8, NG=1      ! 2^3 blocks of 8^3 cells, one ghost layer, on [0, 1]^3
real(R_P),    parameter :: H=1._R_P / (NB * NC)   ! cell spacing
real(R_P),    parameter :: PI=acos(-1._R_P), FACTOR=1._R_P
type(elisha_comm_object)        :: comm
type(elisha_mesh_object)        :: mesh
type(elisha_solver_object)      :: solver
type(elisha_diagnostics_object) :: diag
integer(I4P)              :: coords(4,NB**3), stat, b, i, j, k, n
real(R_P),    allocatable :: f(:), phi(:,:,:,:)
real(R_P)                 :: x(3), lap, res, err, fmax
character(:), allocatable :: errmsg
!region mesh
call comm%init(0, stat, errmsg) ! the caller's communicator (MPI_COMM_WORLD); the serial build ignores it
call check
b = 0
do k=0, NB-1 ; do j=0, NB-1 ; do i=0, NB-1  ! this rank's blocks: (i, j, k) at level 0, then the level
   b = b + 1 ; coords(:,b) = [i, j, k, 0]
enddo ; enddo ; enddo
call mesh%init(nijk=[NC, NC, NC], ngc=NG, nblk0=[NB, NB, NB], dx0=[H, H, H], emin=[0._R_P, 0._R_P, 0._R_P], &
               periodic=[.true., .true., .true.], null=[.false., .false., .false.], coords=coords, stat=stat, &
               errmsg=errmsg)
call check
!endregion mesh
!region source
! the source packed block-major, i fastest: f = sin(2 pi x) sin(2 pi y) sin(2 pi z) at the cell centres
allocate(f(mesh%nb * NC**3), phi(1-NG:NC+NG,1-NG:NC+NG,1-NG:NC+NG,mesh%nb))
n = 0
do b=1, mesh%nb ; do k=1, NC ; do j=1, NC ; do i=1, NC
   n = n + 1 ; x = (mesh%coords(1:3,b) * NC + [i, j, k] - 0.5_R_P) * H
   f(n) = product(sin(2._R_P * PI * x))
enddo ; enddo ; enddo ; enddo
!endregion source
!region solve
call solver%setup(mesh, comm, stat, errmsg)                  ! path auto: one level, so the spectral solver
call check
call solver%solve_packed(mesh, f, FACTOR, phi, stat, errmsg) ! L_h phi = factor (f - mean f), ghosts filled
call check
diag = solver%diagnostics()
print '(A,L1,A,I0)', 'spectral path ', solver%path() == ELISHA_PATH_SPECTRAL, ', iterations ', diag%iterations
!endregion solve
!region check
! the 7-point Laplacian of phi, read through the ghost layers the solver filled, against factor f (mean f is 0);
! and phi against the continuum solution -factor f / (12 pi^2): second order in H
res = 0._R_P ; err = 0._R_P ; fmax = maxval(abs(f)) ; n = 0
do b=1, mesh%nb ; do k=1, NC ; do j=1, NC ; do i=1, NC
   n = n + 1
   lap = (phi(i+1,j,k,b) + phi(i-1,j,k,b) + phi(i,j+1,k,b) + phi(i,j-1,k,b) + phi(i,j,k+1,b) + phi(i,j,k-1,b) &
          - 6._R_P * phi(i,j,k,b)) / H**2
   res = max(res, abs(lap - FACTOR * f(n)))
   err = max(err, abs(phi(i,j,k,b) + FACTOR * f(n) / (12._R_P * PI**2)))
enddo ; enddo ; enddo ; enddo
print '(A,L1)', 'residual below 1e-12 of the source: ', res <= 1.e-12_R_P * FACTOR * fmax
print '(A,ES8.2)', 'max |phi - phi_exact| / max |phi_exact|: ', err / (FACTOR * fmax / (12._R_P * PI**2))
!endregion check
call solver%destroy
call mesh%destroy
call comm%destroy(stat)
contains
   subroutine check
   !< Stop on a failed call, with the message (a program's choice: the library never stops).
   if (stat /= ELISHA_OK) then
      print '(A)', errmsg
      error stop 1
   endif
   endsubroutine check
endprogram quickstart
