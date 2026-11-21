!< Tutorial 7: the same program on one rank or many. The leaves are dealt round-robin to the ranks; each rank gives
!< the solver its own blocks only, and the solver does the rest over the communicator. Built against the serial
!< library (-DELISHA_NO_MPI) it is a plain program; against the MPI library it runs under mpirun.
program t07_mpi
#ifndef ELISHA_NO_MPI
use mpi
#endif
use elisha
implicit none
integer(I4P), parameter :: NB=4, NC=8, NG=1      ! 4^3 level-0 blocks of 8^3 cells on [0, 1]^3
real(R_P),    parameter :: H=1._R_P / (NB * NC), PI=acos(-1._R_P)
type(elisha_comm_object)  :: comm
integer(I4P)              :: stat, run
integer                   :: world, ierr
character(:), allocatable :: errmsg
character(32)             :: arg
logical                   :: bits

call get_command_argument(1, arg) ; bits = trim(arg) == 'bits' ! print the bit pattern of max phi too
#ifndef ELISHA_NO_MPI
call MPI_Init(ierr)
world = MPI_COMM_WORLD
#else
world = 0 ; ierr = 0                     ! the serial build ignores the communicator
#endif
call comm%init(world, stat, errmsg)      ! ELISHA works on its own duplicate of it
call check
if (comm%rank == 0) print '(A,I0)', 'ranks ', comm%nranks
do run=1, 2
   call solve(amr=run == 2)
enddo
call comm%destroy(stat)
#ifndef ELISHA_NO_MPI
call MPI_Finalize(ierr)
#endif
contains
   subroutine solve(amr)
   !< The periodic problem of tutorial 1 on 4^3 blocks, block (0,0,0) refined when amr; this rank's leaves only.
   logical, intent(in) :: amr !< Refine one block (the hierarchical path).
   type(elisha_mesh_object)        :: mesh
   type(elisha_solver_object)      :: solver
   type(elisha_diagnostics_object) :: diag
   integer(I4P)              :: leaves(4,NB**3+7), mine(4,NB**3+7), nleaf, nmine, b, i, j, k, o, n, counts(2)
   real(R_P),    allocatable :: f(:), phi(:,:,:,:)
   real(R_P)                 :: x(3), hb, energy, err(1), phimax(1)
   ! every leaf of the mesh, in a fixed order
   nleaf = 0
   do k=0, NB-1 ; do j=0, NB-1 ; do i=0, NB-1
      if (amr .and. all([i, j, k] == 0)) then
         do o=0, 7
            nleaf = nleaf + 1 ; leaves(:,nleaf) = [ibits(o, 0, 1), ibits(o, 1, 1), ibits(o, 2, 1), 1]
         enddo
      else
         nleaf = nleaf + 1 ; leaves(:,nleaf) = [i, j, k, 0]
      endif
   enddo ; enddo ; enddo
   ! leaf n goes to rank mod(n - 1, nranks): each rank describes its own blocks, and only those
   nmine = 0
   do n=1, nleaf
      if (mod(n - 1, comm%nranks) == comm%rank) then
         nmine = nmine + 1 ; mine(:,nmine) = leaves(:,n)
      endif
   enddo
   call mesh%init(nijk=[NC, NC, NC], ngc=NG, nblk0=[NB, NB, NB], dx0=[H, H, H], emin=[0._R_P, 0._R_P, 0._R_P], &
                  periodic=[.true., .true., .true.], null=[.false., .false., .false.], coords=mine(:,1:nmine), &
                  stat=stat, errmsg=errmsg)
   call check
   allocate(f(mesh%nb * NC**3), phi(1-NG:NC+NG,1-NG:NC+NG,1-NG:NC+NG,mesh%nb))
   n = 0
   do b=1, mesh%nb
      hb = H / 2**mesh%coords(4,b)
      do k=1, NC ; do j=1, NC ; do i=1, NC
         n = n + 1 ; x = (mesh%coords(1:3,b) * NC + [i, j, k] - 0.5_R_P) * hb
         f(n) = product(sin(2._R_P * PI * x))
      enddo ; enddo ; enddo
   enddo
   ! collective: every rank calls setup and solve_packed with its own blocks; the energy comes back on every rank
   call solver%setup(mesh, comm, stat, errmsg, tolerance=1.e-10_R_P)
   call check
   call solver%solve_packed(mesh, f, 1._R_P, phi, stat, errmsg, energy=energy)
   call check
   diag = solver%diagnostics()
   ! the error against -f / (12 pi^2) and max phi, reduced over the ranks; the leaves per rank
   err = 0._R_P ; n = 0
   do b=1, mesh%nb ; do k=1, NC ; do j=1, NC ; do i=1, NC
      n = n + 1 ; err(1) = max(err(1), abs(phi(i,j,k,b) + f(n) / (12._R_P * PI**2)))
   enddo ; enddo ; enddo ; enddo
   call comm%max_real8(err, stat) ; call check
   phimax = maxval(phi(1:NC,1:NC,1:NC,:))      ! a maximum is exact in any order: the same bits for the same phi
   call comm%max_real8(phimax, stat) ; call check
   counts = [mesh%nb, -mesh%nb]
   call comm%max_i4(counts, stat) ; call check
   if (comm%rank == 0) then
      print '(A,A,I3,A,I3,A,I3,A,ES10.3,A,ES9.2)', merge('hierarchical', 'spectral    ', solver%is_hierarchical()), &
            ': leaves per rank', -counts(2), ' to', counts(1), ', FCG', diag%iterations, ', energy ', energy,      &
            ', max error', err(1) * 12._R_P * PI**2
      if (bits) print '(A,Z16.16)', '   max phi, bits ', transfer(phimax(1), 0_I8P)
   endif
   call solver%destroy
   call mesh%destroy
   endsubroutine solve

   subroutine check
   !< Stop on a failed call, with the message.
   if (stat /= ELISHA_OK) then
      print '(A)', errmsg
      error stop 1
   endif
   endsubroutine check
endprogram t07_mpi
