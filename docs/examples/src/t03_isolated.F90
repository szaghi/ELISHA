!< Tutorial 3: a star in empty space. The free-space potential of a Gaussian density, by both paths (Hockney and
!< Eastwood on the spectral one, the James step on the hierarchical one), against the analytic potential; and the
!< periodic solve of the same density, which is the potential of an infinite lattice of stars.
program t03_isolated
!run t03_isolated t03_isolated
use elisha
implicit none
integer(I4P), parameter :: NB=4, NC=8, NG=1      ! 4^3 blocks of 8^3 cells on [0, 1]^3
real(R_P),    parameter :: H=1._R_P / (NB * NC)   ! cell spacing
real(R_P),    parameter :: PI=acos(-1._R_P), FOUR_PI_G=4._R_P * PI, SIGMA=0.07_R_P, C=0.5_R_P
type(elisha_comm_object)   :: comm
type(elisha_mesh_object)   :: mesh
type(elisha_solver_object) :: solver
integer(I4P)              :: coords(4,NB**3), stat, b, i, j, k, n, run
real(R_P),    allocatable :: f(:), phi(:,:,:,:,:)
real(R_P)                 :: x(3), mass, err(3)
character(:), allocatable :: errmsg
character(32)             :: arg
logical                   :: dump

call get_command_argument(1, arg) ; dump = trim(arg) == 'dump' ! the figures' data (docs/scripts), not shown
call comm%init(0, stat, errmsg) ; call check
b = 0
do k=0, NB-1 ; do j=0, NB-1 ; do i=0, NB-1
   b = b + 1 ; coords(:,b) = [i, j, k, 0]
enddo ; enddo ; enddo
!region mesh
! isolated boundaries need every axis non-periodic and cubic cells
call mesh%init(nijk=[NC, NC, NC], ngc=NG, nblk0=[NB, NB, NB], dx0=[H, H, H], emin=[0._R_P, 0._R_P, 0._R_P], &
               periodic=[.false., .false., .false.], null=[.false., .false., .false.], coords=coords, stat=stat, &
               errmsg=errmsg)
call check
!endregion mesh
!region source
! a Gaussian star at the centre, 7 sigma inside every face; its mass is the discrete sum
allocate(f(mesh%nb * NC**3), phi(1-NG:NC+NG,1-NG:NC+NG,1-NG:NC+NG,mesh%nb,3))
n = 0
do b=1, mesh%nb ; do k=1, NC ; do j=1, NC ; do i=1, NC
   n = n + 1 ; x = (mesh%coords(1:3,b) * NC + [i, j, k] - 0.5_R_P) * H
   f(n) = exp(-sum((x - C)**2) / (2._R_P * SIGMA**2))
enddo ; enddo ; enddo ; enddo
mass = sum(f) * H**3
!endregion source
!region solve
! L_h phi = 4 pi G rho with phi -> 0 at infinity (G = 1): on a uniform mesh, auto chooses the spectral path
call solver%setup(mesh, comm, stat, errmsg, isolated=.true.)
call check
call solver%solve_packed(mesh, f, FOUR_PI_G, phi(:,:,:,:,1), stat, errmsg)
call check
! the same with the hierarchical path forced: the James step, two multigrid solves and a boundary convolution
call solver%setup(mesh, comm, stat, errmsg, isolated=.true., path=ELISHA_PATH_HIERARCHICAL, tolerance=1.e-12_R_P)
call check
call solver%solve_packed(mesh, f, FOUR_PI_G, phi(:,:,:,:,2), stat, errmsg)
call check
!endregion solve
!region periodic
! and the periodic solve of the same star: the potential of a lattice of stars, mean density removed
call solver%destroy
call mesh%destroy
call mesh%init(nijk=[NC, NC, NC], ngc=NG, nblk0=[NB, NB, NB], dx0=[H, H, H], emin=[0._R_P, 0._R_P, 0._R_P], &
               periodic=[.true., .true., .true.], null=[.false., .false., .false.], coords=coords, stat=stat, &
               errmsg=errmsg)
call check
call solver%setup(mesh, comm, stat, errmsg)
call check
call solver%solve_packed(mesh, f, FOUR_PI_G, phi(:,:,:,:,3), stat, errmsg)
call check
!endregion periodic
!region compare
! the analytic potential of the Gaussian is -M erf(r / (sqrt(2) sigma)) / r; the max error over the cells, relative to
! the potential at the centre
err = 0._R_P
do run=1, 3
   do b=1, mesh%nb ; do k=1, NC ; do j=1, NC ; do i=1, NC
      x = (mesh%coords(1:3,b) * NC + [i, j, k] - 0.5_R_P) * H
      err(run) = max(err(run), abs(phi(i,j,k,b,run) - potential(norm2(x - C))))
   enddo ; enddo ; enddo ; enddo
enddo
err = err / abs(potential(0._R_P))
print '(A,ES9.2)', 'isolated, spectral path:     max error ', err(1)
print '(A,ES9.2)', 'isolated, hierarchical path: max error ', err(2)
print '(A,ES9.2)', 'periodic:                    max error ', err(3)
print '(A,L1)', 'the two isolated paths agree to 1e-10: ', &
                maxval(abs(phi(1:NC,1:NC,1:NC,:,1) - phi(1:NC,1:NC,1:NC,:,2))) <= 1.e-10_R_P * abs(potential(0._R_P))
!endregion compare
if (dump) then ! phi along the x axis through the centre, every cell of the line: x, exact, the three runs
   open(newunit=n, file='t03_isolated-line.dat', status='replace', action='write')
   do b=1, mesh%nb
      if (any(mesh%coords(2:3,b) /= NB/2 - 1)) cycle ! the blocks holding cell (j, k) = (NC, NC): y = z = C - H/2
      do i=1, NC
         x = ([mesh%coords(1,b) * NC + i, NB/2 * NC, NB/2 * NC] - 0.5_R_P) * H
         write(n, '(5ES24.16)') x(1), potential(norm2(x - C)), phi(i,NC,NC,b,:)
      enddo
   enddo
   close(n)
   call dump_slice('t03_isolated-slice.dat', 1)
endif
call solver%destroy
call mesh%destroy
call comm%destroy(stat)
contains
   pure function potential(r) result(p)
   !< The analytic potential of the Gaussian star of the discrete mass, G = 1.
   real(R_P), intent(in) :: r !< Distance from the centre.
   real(R_P)             :: p !< Potential.
   if (r < 1.e-12_R_P) then
      p = -mass * sqrt(2._R_P / PI) / SIGMA
   else
      p = -mass * erf(r / (sqrt(2._R_P) * SIGMA)) / r
   endif
   endfunction potential

   subroutine dump_slice(file, run)
   !< The plane z = C - H/2 of one run (docs/scripts): per block, its level, lower corner, spacing, then its values.
   character(*), intent(in) :: file !< Output file.
   integer(I4P), intent(in) :: run  !< Run.
   integer(I4P) :: u
   open(newunit=u, file=file, status='replace', action='write')
   do b=1, mesh%nb
      if (mesh%coords(3,b) /= NB/2 - 1) cycle
      write(u, '(I3,3ES24.16)') 0, mesh%coords(1:2,b) * NC * H, H
      write(u, '(*(ES24.16))') phi(1:NC,1:NC,NC,b,run)
   enddo
   close(u)
   endsubroutine dump_slice

   subroutine check
   !< Stop on a failed call, with the message.
   if (stat /= ELISHA_OK) then
      print '(A)', errmsg
      error stop 1
   endif
   endsubroutine check
endprogram t03_isolated
