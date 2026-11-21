!< Cookbook: one setup, many solves. A source that moves a little per time step, solved on one setup (each solve
!< starts from the previous potential) and on a fresh setup per step (each starts from zero): the FCG iterations.
program cookbook_reuse
use elisha
implicit none
integer(I4P), parameter :: NB=4, NC=8, NG=1, NSTEP=5   ! 4^3 level-0 blocks of 8^3 cells, block (0,0,0) refined
real(R_P),    parameter :: H=1._R_P / (NB * NC), PI=acos(-1._R_P), SHIFT=0.01_R_P ! the source moves by SHIFT a step
type(elisha_comm_object)        :: comm
type(elisha_mesh_object)        :: mesh
type(elisha_solver_object)      :: warm, cold
type(elisha_diagnostics_object) :: dw, dc
integer(I4P)              :: coords(4,NB**3+7), stat, nleaf, b, i, j, k, o, n, step
real(R_P),    allocatable :: f(:), phi_warm(:,:,:,:), phi_cold(:,:,:,:)
real(R_P)                 :: x(3), hb
character(:), allocatable :: errmsg

call comm%init(0, stat, errmsg) ; call check
nleaf = 0
do k=0, NB-1 ; do j=0, NB-1 ; do i=0, NB-1
   if (all([i, j, k] == 0)) then
      do o=0, 7
         nleaf = nleaf + 1 ; coords(:,nleaf) = [ibits(o, 0, 1), ibits(o, 1, 1), ibits(o, 2, 1), 1]
      enddo
   else
      nleaf = nleaf + 1 ; coords(:,nleaf) = [i, j, k, 0]
   endif
enddo ; enddo ; enddo
call mesh%init(nijk=[NC, NC, NC], ngc=NG, nblk0=[NB, NB, NB], dx0=[H, H, H], emin=[0._R_P, 0._R_P, 0._R_P], &
               periodic=[.true., .true., .true.], null=[.false., .false., .false.], coords=coords(:,1:nleaf), &
               stat=stat, errmsg=errmsg)
call check
allocate(f(mesh%nb * NC**3))
allocate(phi_warm(1-NG:NC+NG,1-NG:NC+NG,1-NG:NC+NG,mesh%nb), phi_cold(1-NG:NC+NG,1-NG:NC+NG,1-NG:NC+NG,mesh%nb))
call warm%setup(mesh, comm, stat, errmsg, tolerance=1.e-10_R_P) ! once: the mesh does not change
call check
print '(A)', '  step   FCG, one setup   FCG, a setup per step'
do step=1, NSTEP
   call source(step)
   call warm%solve_packed(mesh, f, 1._R_P, phi_warm, stat, errmsg) ! starts from the previous step's potential
   call check
   call cold%setup(mesh, comm, stat, errmsg, tolerance=1.e-10_R_P) ! starts from zero
   call check
   call cold%solve_packed(mesh, f, 1._R_P, phi_cold, stat, errmsg)
   call check
   dw = warm%diagnostics() ; dc = cold%diagnostics()
   print '(I6,I14,I21)', step, dw%iterations, dc%iterations
enddo
call warm%destroy
call cold%destroy
call mesh%destroy
call comm%destroy(stat)
contains
   subroutine source(step)
   !< The source of this step: a Gaussian pulse moving along the diagonal, mean removed by the solver (periodic).
   integer(I4P), intent(in) :: step !< Time step.
   real(R_P) :: centre(3)
   centre = 0.3_R_P + SHIFT * step
   n = 0
   do b=1, mesh%nb
      hb = H / 2**mesh%coords(4,b)
      do k=1, NC ; do j=1, NC ; do i=1, NC
         n = n + 1 ; x = (mesh%coords(1:3,b) * NC + [i, j, k] - 0.5_R_P) * hb
         f(n) = exp(-sum((x - centre)**2) / (2 * 0.06_R_P**2))
      enddo ; enddo ; enddo
   enddo
   endsubroutine source

   subroutine check
   !< Stop on a failed call, with the message.
   if (stat /= ELISHA_OK) then
      print '(A)', errmsg
      error stop 1
   endif
   endsubroutine check
endprogram cookbook_reuse
