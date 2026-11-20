!< Errors: every call returns a status and a message, the library never stops; the caller decides.
program errors
use elisha
implicit none
integer(I4P), parameter :: NB=2, NC=8, NG=1      ! 2^3 level-0 blocks of 8^3 cells on [0, 1]^3
real(R_P),    parameter :: H=1._R_P / (NB * NC)   ! cell spacing at level 0
type(elisha_comm_object)   :: comm
type(elisha_mesh_object)   :: mesh
type(elisha_solver_object) :: solver
integer(I4P)              :: coords(4,NB**3+7), stat, b, i, j, k, o
real(R_P),    allocatable :: f(:), phi(:,:,:,:)
character(:), allocatable :: errmsg
call comm%init(0, stat, errmsg)
b = 0 ! a 2:1 mesh: block (0,0,0) refined to level 1
do k=0, NB-1 ; do j=0, NB-1 ; do i=0, NB-1
   if (all([i, j, k] == 0)) then
      do o=0, 7
         b = b + 1 ; coords(:,b) = [ibits(o, 0, 1), ibits(o, 1, 1), ibits(o, 2, 1), 1]
      enddo
   else
      b = b + 1 ; coords(:,b) = [i, j, k, 0]
   endif
enddo ; enddo ; enddo
call mesh%init(nijk=[NC, NC, NC], ngc=NG, nblk0=[NB, NB, NB], dx0=[H, H, H], emin=[0._R_P, 0._R_P, 0._R_P], &
               periodic=[.true., .true., .true.], null=[.false., .false., .false.], coords=coords, stat=stat, &
               errmsg=errmsg)
allocate(f(mesh%nb * NC**3), phi(1-NG:NC+NG,1-NG:NC+NG,1-NG:NC+NG,mesh%nb))
f = 1._R_P ; phi = 0._R_P
! a solve before any setup
call solver%solve_packed(mesh, f, 1._R_P, phi, stat, errmsg)
call report
! the spectral path asked on a mesh at two levels: refused, and the solver is left not set up
call solver%setup(mesh, comm, stat, errmsg, path=ELISHA_PATH_SPECTRAL)
call report
print '(A,I0)', 'path after the refusal: ', solver%path()
! the caller's choice: fall back to the automatic path
call solver%setup(mesh, comm, stat, errmsg, path=ELISHA_PATH_AUTO)
call report
call solver%destroy
call mesh%destroy
call comm%destroy(stat)
contains
   subroutine report
   !< The status: its code, its name and meaning (elisha_status), and the message of this failure.
   if (stat == ELISHA_OK) then
      print '(A)', 'ok'
   else
      print '(A,I0,A)', 'stat ', stat, ': '//elisha_status_name(stat)//' ('//elisha_status_text(stat)//')'
      print '(A)', '  '//errmsg
   endif
   endsubroutine report
endprogram errors
