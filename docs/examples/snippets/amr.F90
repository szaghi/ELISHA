!< AMR: a 2:1 mesh (one level-0 block refined), the hierarchical path chosen by itself, its diagnostics.
program amr
use elisha
implicit none
integer(I4P), parameter :: NB=2, NC=8, NG=1      ! 2^3 level-0 blocks of 8^3 cells on [0, 1]^3
real(R_P),    parameter :: H=1._R_P / (NB * NC)   ! cell spacing at level 0
real(R_P),    parameter :: PI=acos(-1._R_P), FACTOR=1._R_P
type(elisha_comm_object)        :: comm
type(elisha_mesh_object)        :: mesh
type(elisha_solver_object)      :: solver
type(elisha_diagnostics_object) :: diag
integer(I4P)              :: coords(4,NB**3+7), stat, b, i, j, k, o, n
real(R_P),    allocatable :: f(:), phi(:,:,:,:)
real(R_P)                 :: x(3), h_b, err, emax, energy
character(:), allocatable :: errmsg
call comm%init(0, stat, errmsg) ; call check
! block (0,0,0) is replaced by its 8 children at level 1, coordinates (2i+o_x, 2j+o_y, 2k+o_z) at that level
b = 0
do k=0, NB-1 ; do j=0, NB-1 ; do i=0, NB-1
   if (all([i, j, k] == 0)) then
      do o=0, 7
         b = b + 1 ; coords(:,b) = [2*i + ibits(o, 0, 1), 2*j + ibits(o, 1, 1), 2*k + ibits(o, 2, 1), 1]
      enddo
   else
      b = b + 1 ; coords(:,b) = [i, j, k, 0]
   endif
enddo ; enddo ; enddo
call mesh%init(nijk=[NC, NC, NC], ngc=NG, nblk0=[NB, NB, NB], dx0=[H, H, H], emin=[0._R_P, 0._R_P, 0._R_P], &
               periodic=[.true., .true., .true.], null=[.false., .false., .false.], coords=coords, stat=stat, &
               errmsg=errmsg)
call check
! the source at the cell centres of each block's level: f = sin(2 pi x) sin(2 pi y) sin(2 pi z)
allocate(f(mesh%nb * NC**3), phi(1-NG:NC+NG,1-NG:NC+NG,1-NG:NC+NG,mesh%nb))
n = 0
do b=1, mesh%nb
   h_b = H / 2**mesh%coords(4,b)
   do k=1, NC ; do j=1, NC ; do i=1, NC
      n = n + 1 ; x = (mesh%coords(1:3,b) * NC + [i, j, k] - 0.5_R_P) * h_b
      f(n) = product(sin(2._R_P * PI * x))
   enddo ; enddo ; enddo
enddo
call solver%setup(mesh, comm, stat, errmsg, tolerance=1.e-10_R_P) ! path auto: two levels, so hierarchical
call check
call solver%solve_packed(mesh, f, FACTOR, phi, stat, errmsg, energy=energy) ! energy: sum_cells f phi dV
call check
diag = solver%diagnostics()
print '(A,L1,A,I0,A,L1)', 'hierarchical ', solver%is_hierarchical(), ', FCG iterations ', diag%iterations, &
                          ', converged ', diag%residual <= diag%tolerance_used
print '(A,ES11.4)', 'energy ', energy
! phi against the continuum solution -factor f / (12 pi^2): second order, first order at the 2:1 seam
err = 0._R_P ; emax = FACTOR / (12._R_P * PI**2)
do b=1, mesh%nb ; n = (b - 1) * NC**3
   do k=1, NC ; do j=1, NC ; do i=1, NC
      n = n + 1 ; err = max(err, abs(phi(i,j,k,b) + FACTOR * f(n) / (12._R_P * PI**2)))
   enddo ; enddo ; enddo
enddo
print '(A,ES8.2)', 'max |phi - phi_exact| / max |phi_exact|: ', err / emax
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
endprogram amr
