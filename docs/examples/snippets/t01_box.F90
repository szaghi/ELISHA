!< Tutorial 1: a periodic box. One spectral solve, its residual through the ghost layers, and the second-order
!< convergence of the error against the continuum solution.
program t01_box
use elisha
implicit none
real(R_P), parameter :: PI=acos(-1._R_P)
integer(I4P)              :: nc, level, stat
real(R_P)                 :: res, err, err_prev
character(:), allocatable :: errmsg
character(32)             :: arg
logical                   :: dump
type(elisha_comm_object)  :: comm

call get_command_argument(1, arg) ; dump = trim(arg) == 'dump' ! the figures' data (docs/scripts), not shown
call comm%init(0, stat, errmsg) ! the caller's communicator (MPI_COMM_WORLD); the serial build ignores it
call check
! the same problem on 2^3 blocks of nc^3 cells, nc = 4, 8, 16, 32: the cell spacing halves at every level
print '(A)', '   cells   residual < 1e-12 |f|   max error   order'
err_prev = 0._R_P
do level=0, 3
   nc = 4 * 2**level
   call solve_box(nc, res, err)
   if (level == 0) then
      print '(I5,A,L12,ES20.3)', 2*nc, '^3', res < 1.e-12_R_P, err
   else
      print '(I5,A,L12,ES20.3,F8.2)', 2*nc, '^3', res < 1.e-12_R_P, err, log(err_prev / err) / log(2._R_P)
   endif
   err_prev = err
enddo
call comm%destroy(stat)
contains
   subroutine solve_box(nc, res, err)
   !< Solve L_h phi = f on [0, 1]^3, periodic, with 2^3 blocks of nc^3 cells; return the residual and the error.
   integer(I4P), intent(in)  :: nc  !< Cells per block per axis.
   real(R_P),    intent(out) :: res !< Max residual of the 7-point Laplacian, relative to max |f|.
   real(R_P),    intent(out) :: err !< Max |phi - phi_exact|, relative to max |phi_exact|.
   integer(I4P), parameter         :: NB=2, NG=1          ! 2 blocks per axis, one ghost layer
   real(R_P)                       :: h, x(3), lap
   integer(I4P)                    :: coords(4,NB**3), b, i, j, k, n
   real(R_P), allocatable          :: f(:), phi(:,:,:,:)
   type(elisha_mesh_object)        :: mesh
   type(elisha_solver_object)      :: solver
   h = 1._R_P / (NB * nc)                    ! cell spacing
   b = 0
   do k=0, NB-1 ; do j=0, NB-1 ; do i=0, NB-1 ! this rank's blocks: (i, j, k) at level 0, then the level
      b = b + 1 ; coords(:,b) = [i, j, k, 0]
   enddo ; enddo ; enddo
   call mesh%init(nijk=[nc, nc, nc], ngc=NG, nblk0=[NB, NB, NB], dx0=[h, h, h], emin=[0._R_P, 0._R_P, 0._R_P], &
                  periodic=[.true., .true., .true.], null=[.false., .false., .false.], coords=coords, stat=stat, &
                  errmsg=errmsg)
   call check
   ! the source packed block-major, i fastest: f = sin(2 pi x) sin(2 pi y) sin(2 pi z) at the cell centres
   allocate(f(mesh%nb * nc**3), phi(1-NG:nc+NG,1-NG:nc+NG,1-NG:nc+NG,mesh%nb))
   n = 0
   do b=1, mesh%nb ; do k=1, nc ; do j=1, nc ; do i=1, nc
      n = n + 1 ; x = (mesh%coords(1:3,b) * nc + [i, j, k] - 0.5_R_P) * h
      f(n) = product(sin(2._R_P * PI * x))
   enddo ; enddo ; enddo ; enddo
   call solver%setup(mesh, comm, stat, errmsg)              ! path auto: one level, so the spectral solver
   call check
   call solver%solve_packed(mesh, f, 1._R_P, phi, stat, errmsg) ! L_h phi = f - mean f, every ghost layer filled
   call check
   ! the 7-point Laplacian of phi, read through the ghost layers the solver filled, against f (mean f is 0); and phi
   ! against the continuum solution -f / (12 pi^2)
   res = 0._R_P ; err = 0._R_P ; n = 0
   do b=1, mesh%nb ; do k=1, nc ; do j=1, nc ; do i=1, nc
      n = n + 1
      lap = (phi(i+1,j,k,b) + phi(i-1,j,k,b) + phi(i,j+1,k,b) + phi(i,j-1,k,b) + phi(i,j,k+1,b) + phi(i,j,k-1,b) &
             - 6._R_P * phi(i,j,k,b)) / h**2
      res = max(res, abs(lap - f(n)))
      err = max(err, abs(phi(i,j,k,b) + f(n) / (12._R_P * PI**2)))
   enddo ; enddo ; enddo ; enddo
   res = res / maxval(abs(f)) ; err = err * 12._R_P * PI**2 / maxval(abs(f))
   if (dump .and. nc == 16) call dump_slice('t01_box-slice.dat', mesh, phi, 0.5_R_P - 0.5_R_P * h)
   call solver%destroy
   call mesh%destroy
   endsubroutine solve_box

   subroutine dump_slice(file, mesh, phi, z)
   !< Write the cells of the plane through z of every block crossing it: per block, its level, its lower corner and
   !< spacing, then its nc x nc values of phi (the figures of docs/scripts read it).
   character(*),             intent(in) :: file           !< Output file.
   type(elisha_mesh_object), intent(in) :: mesh           !< Mesh.
   real(R_P),                intent(in) :: phi(1-mesh%ngc:,1-mesh%ngc:,1-mesh%ngc:,:) !< Potential.
   real(R_P),                intent(in) :: z              !< Height of the plane.
   integer(I4P) :: u, b, k
   real(R_P)    :: hb
   open(newunit=u, file=file, status='replace', action='write')
   do b=1, mesh%nb
      hb = mesh%dx0(3) / 2**mesh%coords(4,b)
      k = floor((z - mesh%coords(3,b) * mesh%nijk(3) * hb) / hb) + 1
      if (k < 1 .or. k > mesh%nijk(3)) cycle
      write(u, '(I3,3ES24.16)') mesh%coords(4,b), mesh%coords(1:2,b) * mesh%nijk(1:2) * hb, hb
      write(u, '(*(ES24.16))') phi(1:mesh%nijk(1),1:mesh%nijk(2),k,b)
   enddo
   close(u)
   endsubroutine dump_slice

   subroutine check
   !< Stop on a failed call, with the message (a program's choice: the library never stops).
   if (stat /= ELISHA_OK) then
      print '(A)', errmsg
      error stop 1
   endif
   endsubroutine check
endprogram t01_box
