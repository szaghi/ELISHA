!< Isolated boundaries: the free-space potential of a Gaussian, against the analytic one, ghost layers included.
program isolated
!run isolated isolated
use elisha
implicit none
integer(I4P), parameter :: NB=4, NC=8, NG=2      ! 4^3 blocks of 8^3 cells on [0, 1]^3, two ghost layers
real(R_P),    parameter :: H=1._R_P / (NB * NC)   ! cell spacing
real(R_P),    parameter :: PI=acos(-1._R_P), FACTOR=4._R_P * PI, SIGMA=0.08_R_P, C=0.5_R_P
type(elisha_comm_object)   :: comm
type(elisha_mesh_object)   :: mesh
type(elisha_solver_object) :: solver
integer(I4P)              :: coords(4,NB**3), stat, b, i, j, k, n, ig
real(R_P),    allocatable :: f(:), phi(:,:,:,:)
real(R_P)                 :: x(3), mass, r, exact
character(:), allocatable :: errmsg
call comm%init(0, stat, errmsg) ; call check
b = 0
do k=0, NB-1 ; do j=0, NB-1 ; do i=0, NB-1
   b = b + 1 ; coords(:,b) = [i, j, k, 0]
enddo ; enddo ; enddo
! isolated boundaries: every axis non-periodic, cubic cells
call mesh%init(nijk=[NC, NC, NC], ngc=NG, nblk0=[NB, NB, NB], dx0=[H, H, H], emin=[0._R_P, 0._R_P, 0._R_P], &
               periodic=[.false., .false., .false.], null=[.false., .false., .false.], coords=coords, stat=stat, &
               errmsg=errmsg)
call check
! a Gaussian density at the centre, 6 sigma inside every face; its mass the discrete sum
allocate(f(mesh%nb * NC**3), phi(1-NG:NC+NG,1-NG:NC+NG,1-NG:NC+NG,mesh%nb))
n = 0
do b=1, mesh%nb ; do k=1, NC ; do j=1, NC ; do i=1, NC
   n = n + 1 ; x = (mesh%coords(1:3,b) * NC + [i, j, k] - 0.5_R_P) * H
   f(n) = exp(-sum((x - C)**2) / (2._R_P * SIGMA**2))
enddo ; enddo ; enddo ; enddo
mass = sum(f) * H**3
!region solve
! L_h phi = 4 pi f with phi -> 0 at infinity: Hockney and Eastwood on the spectral path (a uniform mesh)
call solver%setup(mesh, comm, stat, errmsg, isolated=.true.)
call check
call solver%solve_packed(mesh, f, FACTOR, phi, stat, errmsg)
call check
!endregion solve
!region compare
! along the line y = z = 15.5 H, from x = 24.5 H out to the last block's ghost layer i = 34 (outside); the analytic
! potential of the Gaussian is -M erf(r / (sqrt(2) sigma)) / r
print '(A)', '   i       r     phi       exact     rel. error'
b = 1 + (NB - 1) + NB * (NB/2 - 1) + NB**2 * (NB/2 - 1) ! the block of x >= 0.75 holding cell (j, k) = (16, 16)
do ig=(NB - 1)*NC + 1, NB*NC + NG, 3
   i = ig - (NB - 1) * NC
   x = ([ig, 16, 16] - 0.5_R_P) * H
   r = norm2(x - C)
   exact = -mass * erf(r / (sqrt(2._R_P) * SIGMA)) / r
   print '(I4,F8.4,2F10.4,ES11.1)', ig, r, phi(i,NC,NC,b), exact, abs(phi(i,NC,NC,b) / exact - 1._R_P)
enddo
!endregion compare
call solver%destroy
call mesh%destroy
call comm%destroy(stat)
contains
   subroutine check
   !< Stop on a failed call, with the message.
   if (stat /= ELISHA_OK) then
      print '(A)', errmsg
      error stop 1
   endif
   endsubroutine check
endprogram isolated
