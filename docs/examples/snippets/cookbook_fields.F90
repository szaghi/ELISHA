!< Cookbook: your fields in, your fields out. A field with ghost cells packed into the solver's source in one line, the
!< potential returned in the same layout; and a singular problem (periodic), where the mean of the source is removed.
program cookbook_fields
use elisha
implicit none
integer(I4P), parameter :: NB=2, NC=8, NG=2      ! 2^3 blocks of 8^3 cells, two ghost layers (your code's, say)
real(R_P),    parameter :: H=1._R_P / (NB * NC), PI=acos(-1._R_P)
type(elisha_comm_object)   :: comm
type(elisha_mesh_object)   :: mesh
type(elisha_solver_object) :: solver
integer(I4P)               :: coords(4,NB**3), stat, b, i, j, k
real(R_P), allocatable     :: rho(:,:,:,:), phi(:,:,:,:), buf(:)
real(R_P)                  :: x(3), lap, res, mean_rho
character(:), allocatable  :: errmsg

call comm%init(0, stat, errmsg) ; call check
b = 0
do k=0, NB-1 ; do j=0, NB-1 ; do i=0, NB-1
   b = b + 1 ; coords(:,b) = [i, j, k, 0]
enddo ; enddo ; enddo
call mesh%init(nijk=[NC, NC, NC], ngc=NG, nblk0=[NB, NB, NB], dx0=[H, H, H], emin=[0._R_P, 0._R_P, 0._R_P], &
               periodic=[.true., .true., .true.], null=[.false., .false., .false.], coords=coords, stat=stat, &
               errmsg=errmsg)
call check
! your density, ghosts and all: rho(1-NG:NC+NG, 1-NG:NC+NG, 1-NG:NC+NG, blocks); a mean of 1 under a wave
allocate(rho(1-NG:NC+NG,1-NG:NC+NG,1-NG:NC+NG,mesh%nb), phi(1-NG:NC+NG,1-NG:NC+NG,1-NG:NC+NG,mesh%nb))
rho = huge(1._R_P)                                         ! the ghosts: the solver never reads them
do b=1, mesh%nb ; do k=1, NC ; do j=1, NC ; do i=1, NC
   x = (mesh%coords(1:3,b) * NC + [i, j, k] - 0.5_R_P) * H
   rho(i,j,k,b) = 1._R_P + 0.5_R_P * sin(2 * PI * x(1)) * cos(2 * PI * x(2))
enddo ; enddo ; enddo ; enddo
! the interior in Fortran's order is the solver's packed order: block-major, i fastest
buf = reshape(rho(1:NC,1:NC,1:NC,:), [NC**3 * mesh%nb])
call solver%setup(mesh, comm, stat, errmsg)
call check
call solver%solve_packed(mesh, buf, 1._R_P, phi, stat, errmsg) ! periodic: L_h phi = rho - mean rho
call check
! the residual against rho minus its mean, through the ghosts the solver filled (NG = 2: both layers filled)
mean_rho = sum(buf) / size(buf) ; res = 0._R_P
do b=1, mesh%nb ; do k=1, NC ; do j=1, NC ; do i=1, NC
   lap = (phi(i+1,j,k,b) + phi(i-1,j,k,b) + phi(i,j+1,k,b) + phi(i,j-1,k,b) + phi(i,j,k+1,b) + phi(i,j,k-1,b) &
          - 6 * phi(i,j,k,b)) / H**2
   res = max(res, abs(lap - (rho(i,j,k,b) - mean_rho)))
enddo ; enddo ; enddo ; enddo
print '(A,F6.3)', 'mean of rho ', mean_rho
print '(A,L1)', 'L_h phi = rho - mean rho to 1e-12: ', res <= 1.e-12_R_P
print '(A,L1)', 'mean of phi zero to 1e-12: ', abs(sum(phi(1:NC,1:NC,1:NC,:))) / (NC**3 * mesh%nb) <= 1.e-12_R_P
print '(A,L1)', 'second ghost layer filled: ', all(phi(-1,1:NC,1:NC,:) /= 0._R_P)
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
endprogram cookbook_fields
