!< Tutorial 8: two embers. Two Gaussian masses on a circular orbit in free space; the AMR mesh follows them, so every
!< frame is a regrid: a new mesh, a new setup, one isolated solve on the hierarchical path. Two equal masses at the
!< ends of a diameter look the same after half a turn: the frames cover 180 degrees (the animation loops on them), and
!< the plain run solves every 8th frame.
program t08_embers
!run t08_embers t08_embers
use elisha
implicit none
integer(I4P), parameter :: NB=6, NC=8, NG=1, LMAX=2, NFRAME=48 ! 6^3 level-0 blocks of 8^3 cells on [0, 1]^3
real(R_P),    parameter :: H=1._R_P / (NB * NC), PI=acos(-1._R_P), FOUR_PI_G=4._R_P * PI
real(R_P),    parameter :: SIGMA=0.03_R_P, ORBIT=0.11_R_P, C=0.5_R_P ! the embers' width, orbit radius, centre
real(R_P),    parameter :: REACH(0:1)=[0.14_R_P, 0.04_R_P] ! refine a block at level l when an ember is this close
type(elisha_comm_object)        :: comm
type(elisha_mesh_object)        :: mesh
type(elisha_solver_object)      :: solver
type(elisha_diagnostics_object) :: diag
integer(I4P), allocatable :: coords(:,:)
real(R_P),    allocatable :: f(:), phi(:,:,:,:)
integer(I4P)              :: stat, frame, nleaf, b, i, j, k, n, u
real(R_P)                 :: ember(3,2), x(3), hb, energy
character(:), allocatable :: errmsg
character(32)             :: arg
logical                   :: dump

call get_command_argument(1, arg) ; dump = trim(arg) == 'dump' ! the figures' data (docs/scripts), not shown
call comm%init(0, stat, errmsg) ; call check
allocate(coords(4,NB**3 * 8**LMAX))
if (dump) open(newunit=u, file='t08_embers-frames.dat', status='replace', action='write')
print '(A)', '  angle   leaves L0/L1/L2    FCG   energy'
do frame=0, NFRAME-1
   if (.not.dump .and. mod(frame, 8) /= 0) cycle
   !region orbit
   ! the embers at opposite ends of a diameter, turning by pi / NFRAME per frame
   ember(:,1) = C + ORBIT * [cos(PI * frame / NFRAME), sin(PI * frame / NFRAME), 0._R_P]
   ember(:,2) = 2 * C - ember(:,1)
   !endregion orbit
   !region regrid
   ! a new mesh around the embers' new places, and a new setup on it (setup destroys the previous solver)
   nleaf = 0
   do k=0, NB-1 ; do j=0, NB-1 ; do i=0, NB-1
      call refine([i, j, k, 0])
   enddo ; enddo ; enddo
   call mesh%destroy
   call mesh%init(nijk=[NC, NC, NC], ngc=NG, nblk0=[NB, NB, NB], dx0=[H, H, H], emin=[0._R_P, 0._R_P, 0._R_P], &
                  periodic=[.false., .false., .false.], null=[.false., .false., .false.], coords=coords(:,1:nleaf), &
                  stat=stat, errmsg=errmsg)
   call check
   call solver%setup(mesh, comm, stat, errmsg, isolated=.true., tolerance=1.e-10_R_P)
   call check
   !endregion regrid
   if (allocated(f)) deallocate(f, phi)
   allocate(f(mesh%nb * NC**3), phi(1-NG:NC+NG,1-NG:NC+NG,1-NG:NC+NG,mesh%nb))
   n = 0
   do b=1, mesh%nb
      hb = H / 2**mesh%coords(4,b)
      do k=1, NC ; do j=1, NC ; do i=1, NC
         n = n + 1 ; x = (mesh%coords(1:3,b) * NC + [i, j, k] - 0.5_R_P) * hb
         f(n) = exp(-sum((x - ember(:,1))**2) / (2 * SIGMA**2)) + exp(-sum((x - ember(:,2))**2) / (2 * SIGMA**2))
      enddo ; enddo ; enddo
   enddo
   !region solve
   call solver%solve_packed(mesh, f, FOUR_PI_G, phi, stat, errmsg, energy=energy)
   call check
   diag = solver%diagnostics()
   !endregion solve
   if (mod(frame, 8) == 0) print '(I6,A,3I6,I7,ES12.3)', 180 * frame / NFRAME, ' deg', &
                                 (count(mesh%coords(4,:) == i), i=0, 2), diag%iterations, energy
   if (dump) call write_frame
enddo
if (dump) close(u)
call solver%destroy
call mesh%destroy
call comm%destroy(stat)
contains
   recursive subroutine refine(c4)
   !< Emit the block c4 as a leaf, or its 8 children when it is above level LMAX and an ember is within the reach of
   !< its level (REACH(1) + a level-1 block <= REACH(0): the mesh stays 2:1; REACH(0) keeps the boundary at level 0).
   integer(I4P), intent(in) :: c4(4) !< Block (i, j, k, level).
   integer(I4P) :: o
   real(R_P)    :: lo(3), hi(3)
   logical      :: near
   lo = c4(1:3) * NC * H / 2**c4(4) ; hi = lo + NC * H / 2**c4(4)
   near = .false.
   if (c4(4) < LMAX) near = norm2(max(lo - ember(:,1), 0._R_P, ember(:,1) - hi)) < REACH(c4(4)) .or. &
                            norm2(max(lo - ember(:,2), 0._R_P, ember(:,2) - hi)) < REACH(c4(4))
   if (near) then
      do o=0, 7
         call refine([2*c4(1) + ibits(o, 0, 1), 2*c4(2) + ibits(o, 1, 1), 2*c4(3) + ibits(o, 2, 1), c4(4) + 1])
      enddo
   else
      nleaf = nleaf + 1 ; coords(:,nleaf) = c4
   endif
   endsubroutine refine

   subroutine write_frame
   !< The plane z = C (just below it) of this frame (docs/scripts): the embers, then per block crossing it, its level,
   !< lower corner, spacing and values.
   integer(I4P) :: kk, nslice
   nslice = 0
   do b=1, mesh%nb
      hb = H / 2**mesh%coords(4,b)
      kk = floor((C - 0.25_R_P * H / 2**LMAX - mesh%coords(3,b) * NC * hb) / hb) + 1
      if (kk >= 1 .and. kk <= NC) nslice = nslice + 1
   enddo
   write(u, '(I6,4ES24.16,I6)') frame, ember(1:2,1), ember(1:2,2), nslice
   do b=1, mesh%nb
      hb = H / 2**mesh%coords(4,b)
      kk = floor((C - 0.25_R_P * H / 2**LMAX - mesh%coords(3,b) * NC * hb) / hb) + 1
      if (kk < 1 .or. kk > NC) cycle
      write(u, '(I3,3ES24.16)') mesh%coords(4,b), mesh%coords(1:2,b) * NC * hb, hb
      write(u, '(*(ES16.8))') phi(1:NC,1:NC,kk,b)
   enddo
   endsubroutine write_frame

   subroutine check
   !< Stop on a failed call, with the message.
   if (stat /= ELISHA_OK) then
      print '(A)', errmsg
      error stop 1
   endif
   endsubroutine check
endprogram t08_embers
