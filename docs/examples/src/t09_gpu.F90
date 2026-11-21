!< Tutorial 9: on the GPU. The device backends (cuFFT, OpenACC) chosen by molds, and the device-resident API: the
!< source and the potential already on the device (on_device), against the same solve from host arrays. Built
!< without OpenACC (the docs' serial build) the program runs the host solvers, and on_device is off.
program t09_gpu
!run t09_gpu t09_gpu
!region use
#ifndef ELISHA_NO_MPI
use mpi
#endif
use elisha
#ifdef _OPENACC
use elisha_device, only : elisha_spectral_solver_cufft_object, elisha_hierarchical_solver_oac_object
#endif
implicit none
!endregion use
integer(I4P), parameter :: NB=4, NC=16, NG=1     ! 4^3 level-0 blocks of 16^3 cells on [0, 1]^3
real(R_P),    parameter :: H=1._R_P / (NB * NC), PI=acos(-1._R_P)
#ifdef _OPENACC
logical,      parameter :: ON_DEVICE=.true.
#else
logical,      parameter :: ON_DEVICE=.false.
#endif
type(elisha_comm_object)  :: comm
integer(I4P)              :: stat, run
integer                   :: world, ierr
character(:), allocatable :: errmsg

#ifndef ELISHA_NO_MPI
call MPI_Init(ierr)
world = MPI_COMM_WORLD
#else
world = 0 ; ierr = 0
#endif
call comm%init(world, stat, errmsg) ; call check
if (comm%rank == 0) print '(A,L1)', 'device backends and on_device: ', ON_DEVICE
do run=1, 2
   call solve(amr=run == 2)
enddo
call comm%destroy(stat)
#ifndef ELISHA_NO_MPI
call MPI_Finalize(ierr)
#endif
contains
   subroutine solve(amr)
   !< The periodic problem of tutorial 7, 16^3 cells per block, block (0,0,0) refined when amr; leaves round-robin.
   logical, intent(in) :: amr !< Refine one block (the hierarchical path).
   !region molds
#ifdef _OPENACC
   type(elisha_spectral_solver_cufft_object)   :: cufft ! molds: only their types are read
   type(elisha_hierarchical_solver_oac_object) :: oac
#endif
   !endregion molds
   type(elisha_mesh_object)        :: mesh
   type(elisha_solver_object)      :: solver
   type(elisha_diagnostics_object) :: diag
   integer(I4P)              :: leaves(4,NB**3+7), mine(4,NB**3+7), nleaf, nmine, b, i, j, k, o, n, ndiff(1), pass
   real(R_P),    allocatable :: f(:), phi_host(:,:,:,:), phi_dev(:,:,:,:)
   real(R_P)                 :: x(3), hb, err(1)
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
   allocate(f(mesh%nb * NC**3))
   allocate(phi_host(1-NG:NC+NG,1-NG:NC+NG,1-NG:NC+NG,mesh%nb), phi_dev(1-NG:NC+NG,1-NG:NC+NG,1-NG:NC+NG,mesh%nb))
   phi_host = 0._R_P ; phi_dev = 0._R_P ! the hierarchical path leaves the edge and corner ghosts as they are
   n = 0
   do b=1, mesh%nb
      hb = H / 2**mesh%coords(4,b)
      do k=1, NC ; do j=1, NC ; do i=1, NC
         n = n + 1 ; x = (mesh%coords(1:3,b) * NC + [i, j, k] - 0.5_R_P) * hb
         f(n) = product(sin(2._R_P * PI * x))
      enddo ; enddo ; enddo
   enddo
   do pass=1, 2 ! host arrays, then device-resident ones; a new setup for each: a hierarchical solve warm-starts
                ! from the previous solve of its setup
      !region setup
#ifdef _OPENACC
      call solver%setup(mesh, comm, stat, errmsg, tolerance=1.e-10_R_P, spectral_mold=cufft, hierarchical_mold=oac)
#else
      call solver%setup(mesh, comm, stat, errmsg, tolerance=1.e-10_R_P)
#endif
      !endregion setup
      call check
      if (pass == 1) then
         !region host_arrays
         ! host arrays: the backend copies the source in and the potential out
         call solver%solve_packed(mesh, f, 1._R_P, phi_host, stat, errmsg)
         !endregion host_arrays
         diag = solver%diagnostics()
      else
         !region device_arrays
         ! device-resident arrays: f and phi are present on the device, and the solve stages nothing (copy, not
         ! copyout: the ghosts the solve does not write keep their values)
         !$acc data copyin(f) copy(phi_dev)
         call solver%solve_packed(mesh, f, 1._R_P, phi_dev, stat, errmsg, on_device=ON_DEVICE)
         !$acc end data
         !endregion device_arrays
      endif
      call check
   enddo
   err = 0._R_P ; n = 0
   do b=1, mesh%nb ; do k=1, NC ; do j=1, NC ; do i=1, NC
      n = n + 1 ; err(1) = max(err(1), abs(phi_dev(i,j,k,b) + f(n) / (12._R_P * PI**2)))
   enddo ; enddo ; enddo ; enddo
   call comm%max_real8(err, stat) ; call check
   ndiff = count(phi_dev /= phi_host)                             ! every cell and ghost
   call comm%max_i4(ndiff, stat) ; call check
   if (comm%rank == 0) print '(A,A,I3,A,ES9.2,A,I0)', merge('hierarchical', 'spectral    ', solver%is_hierarchical()), &
                             ': FCG', diag%iterations, ', max error', err(1) * 12._R_P * PI**2,                         &
                             ', values differing between on_device and host arrays ', ndiff(1)
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
endprogram t09_gpu
