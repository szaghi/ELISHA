!< Cookbook: which faces go together. Face configurations through setup, each with the status it returns.
program cookbook_bc
!run cookbook_bc cookbook_bc
use elisha
implicit none
integer(I4P), parameter :: NB=2, NC=4, NG=1, P=ELISHA_BC_PERIODIC, D=ELISHA_BC_DIRICHLET, N=ELISHA_BC_NEUMANN
integer(I4P), parameter :: G=ELISHA_BC_DIRICHLET_GHOST
real(R_P),    parameter :: H=1._R_P / (NB * NC)
type(elisha_comm_object)  :: comm
integer(I4P)              :: stat
character(:), allocatable :: errmsg

call comm%init(0, stat, errmsg)
!region cases
!         label                                       periodic mesh axes    x faces  y faces  z faces  isolated
call try('D/N, N/D, D/D on a bounded box          ', [.false., .false., .false.], [D, N], [N, D], [D, D], .false.)
call try('N/N everywhere: singular, mean removed  ', [.false., .false., .false.], [N, N], [N, N], [N, N], .false.)
call try('periodic x, ghost-Dirichlet y, D/N z    ', [.true., .false., .false.], [P, P], [G, G], [D, N], .false.)
call try('a Dirichlet face on a periodic axis     ', [.true., .false., .false.], [D, D], [D, D], [D, D], .false.)
call try('one periodic face, one Dirichlet        ', [.false., .false., .false.], [P, D], [D, D], [D, D], .false.)
call try('ghost-Dirichlet with plain Dirichlet    ', [.false., .false., .false.], [G, D], [D, D], [D, D], .false.)
call try('a bounded box without kinds             ', [.false., .false., .false.], [-1, -1], [-1, -1], [-1, -1], .false.)
call try('isolated, on a bounded box              ', [.false., .false., .false.], [-1, -1], [-1, -1], [-1, -1], .true.)
call try('isolated, with kinds                    ', [.false., .false., .false.], [D, D], [D, D], [D, D], .true.)
call try('isolated, on a periodic axis            ', [.true., .false., .false.], [-1, -1], [-1, -1], [-1, -1], .true.)
!endregion cases
call comm%destroy(stat)
contains
   subroutine try(label, periodic, x, y, z, isolated)
   !< Set up the solver on 2^3 blocks with these faces (kinds -1: none given) and print the status.
   character(*), intent(in) :: label       !< What the case is.
   logical,      intent(in) :: periodic(3) !< The mesh's periodic axes.
   integer(I4P), intent(in) :: x(2)        !< Kinds of the x faces (-1: no bc argument).
   integer(I4P), intent(in) :: y(2)        !< Kinds of the y faces.
   integer(I4P), intent(in) :: z(2)        !< Kinds of the z faces.
   logical,      intent(in) :: isolated    !< Isolated boundaries.
   type(elisha_mesh_object)   :: mesh
   type(elisha_solver_object) :: solver
   integer(I4P)               :: coords(4,NB**3), bc(2,3), b, i, j, k
   b = 0
   do k=0, NB-1 ; do j=0, NB-1 ; do i=0, NB-1
      b = b + 1 ; coords(:,b) = [i, j, k, 0]
   enddo ; enddo ; enddo
   call mesh%init(nijk=[NC, NC, NC], ngc=NG, nblk0=[NB, NB, NB], dx0=[H, H, H], emin=[0._R_P, 0._R_P, 0._R_P], &
                  periodic=periodic, null=[.false., .false., .false.], coords=coords, stat=stat, errmsg=errmsg)
   bc(:,1) = x ; bc(:,2) = y ; bc(:,3) = z
   if (any(bc < 0)) then
      call solver%setup(mesh, comm, stat, errmsg, isolated=isolated)
   else
      call solver%setup(mesh, comm, stat, errmsg, isolated=isolated, bc=bc)
   endif
   print '(A,2X,A)', label, elisha_status_name(stat)
   call solver%destroy
   call mesh%destroy
   endsubroutine try
endprogram cookbook_bc
