!< Tutorial 4: refine where it matters. The star of tutorial 3 on a coarse uniform mesh, on a 2:1 AMR mesh refined
!< twice around it, and on the uniform mesh of the finest spacing: the hierarchical path, its iterations, and what
!< the refinement buys per cell.
program t04_amr
use elisha
implicit none
integer(I4P), parameter :: NB=4, NC=4, NG=1      ! 4^3 level-0 blocks of 4^3 cells on [0, 1]^3
real(R_P),    parameter :: H=1._R_P / (NB * NC)   ! cell spacing at level 0
real(R_P),    parameter :: PI=acos(-1._R_P), FOUR_PI_G=4._R_P * PI, SIGMA=0.05_R_P, C=0.5_R_P
type(elisha_comm_object)        :: comm
type(elisha_mesh_object)        :: mesh
type(elisha_solver_object)      :: solver
type(elisha_diagnostics_object) :: diag
integer(I4P), allocatable :: coords(:,:)
real(R_P),    allocatable :: f(:), phi(:,:,:,:)
integer(I4P)              :: stat, nleaf, b, i, j, k, n, run, lmax
real(R_P)                 :: x(3), hb, mass, err, reach(0:1)=[0.24_R_P, 0.12_R_P] ! refine a block at level l
                                                                                    ! when it comes this close
character(:), allocatable :: errmsg
character(32)             :: arg
character(16), parameter  :: NAMES(3)=['uniform, coarse ', 'AMR, 3 levels   ', 'uniform, fine   ']
logical                   :: dump, everywhere

call get_command_argument(1, arg) ; dump = trim(arg) == 'dump' ! the figures' data (docs/scripts), not shown
if (.not.dump .and. len_trim(arg) > 0) then ! t04_amr R0 R1: the AMR mesh only, refined within other reaches
   read(arg, *) reach(0)
   call get_command_argument(2, arg) ; read(arg, *) reach(1)
endif
call comm%init(0, stat, errmsg) ; call check
print '(A)', '  mesh              leaves L0/L1/L2      cells   path           FCG   max error'
do run=1, 3
   if (len_trim(arg) > 0 .and. .not.dump .and. run /= 2) cycle
   lmax = merge(0, 2, run == 1) ; everywhere = run == 3
   ! the leaves of the octree: a block near the star is replaced by its 8 children, down to level lmax
   allocate(coords(4,NB**3 * 8**lmax)) ; nleaf = 0
   do k=0, NB-1 ; do j=0, NB-1 ; do i=0, NB-1
      call refine([i, j, k, 0])
   enddo ; enddo ; enddo
   call mesh%init(nijk=[NC, NC, NC], ngc=NG, nblk0=[NB, NB, NB], dx0=[H, H, H], emin=[0._R_P, 0._R_P, 0._R_P], &
                  periodic=[.false., .false., .false.], null=[.false., .false., .false.], coords=coords(:,1:nleaf), &
                  stat=stat, errmsg=errmsg)
   call check
   ! the star at the cell centres of each block's own level: a block at level l has spacing H / 2^l
   allocate(f(mesh%nb * NC**3), phi(1-NG:NC+NG,1-NG:NC+NG,1-NG:NC+NG,mesh%nb))
   n = 0 ; mass = 0._R_P
   do b=1, mesh%nb
      hb = H / 2**mesh%coords(4,b)
      do k=1, NC ; do j=1, NC ; do i=1, NC
         n = n + 1 ; x = (mesh%coords(1:3,b) * NC + [i, j, k] - 0.5_R_P) * hb
         f(n) = exp(-sum((x - C)**2) / (2._R_P * SIGMA**2)) ; mass = mass + f(n) * hb**3
      enddo ; enddo ; enddo
   enddo
   ! auto: the spectral path on a uniform mesh, the hierarchical one (multigrid FCG, the James step) on the AMR mesh
   call solver%setup(mesh, comm, stat, errmsg, isolated=.true., tolerance=1.e-10_R_P)
   call check
   call solver%solve_packed(mesh, f, FOUR_PI_G, phi, stat, errmsg)
   call check
   diag = solver%diagnostics()
   err = 0._R_P
   do b=1, mesh%nb
      hb = H / 2**mesh%coords(4,b)
      do k=1, NC ; do j=1, NC ; do i=1, NC
         x = (mesh%coords(1:3,b) * NC + [i, j, k] - 0.5_R_P) * hb
         err = max(err, abs(phi(i,j,k,b) - potential(norm2(x - C))))
      enddo ; enddo ; enddo
   enddo
   print '(2X,A,3I5,I11,3X,A,I5,ES12.2)', NAMES(run), (count(mesh%coords(4,:) == i), i=0, 2), mesh%nb * NC**3, &
         merge('hierarchical', 'spectral    ', solver%is_hierarchical()), diag%iterations, err / abs(potential(0._R_P))
   if (dump .and. run == 2) call dump_slice('t04_amr-slice.dat')
   call solver%destroy
   call mesh%destroy
   deallocate(coords, f, phi)
enddo
call comm%destroy(stat)
contains
   recursive subroutine refine(c4)
   !< Emit the block c4 as a leaf, or its 8 children when it is near the star and above level lmax.
   integer(I4P), intent(in) :: c4(4) !< Block (i, j, k, level).
   integer(I4P) :: o
   real(R_P)    :: lo(3), hi(3)
   lo = c4(1:3) * NC * H / 2**c4(4) ; hi = lo + NC * H / 2**c4(4)
   if (c4(4) < lmax .and. (everywhere .or. norm2(max(lo - C, 0._R_P, C - hi)) < reach(min(c4(4), 1)))) then
      do o=0, 7
         call refine([2*c4(1) + ibits(o, 0, 1), 2*c4(2) + ibits(o, 1, 1), 2*c4(3) + ibits(o, 2, 1), c4(4) + 1])
      enddo
   else
      nleaf = nleaf + 1 ; coords(:,nleaf) = c4
   endif
   endsubroutine refine

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

   subroutine dump_slice(file)
   !< The plane z = C (just below it) of the AMR run (docs/scripts): per block crossing it, its level, lower corner,
   !< spacing, then its values.
   character(*), intent(in) :: file !< Output file.
   integer(I4P) :: u, kk
   open(newunit=u, file=file, status='replace', action='write')
   do b=1, mesh%nb
      hb = H / 2**mesh%coords(4,b)
      kk = floor((C - 0.5_R_P * H / 4 - mesh%coords(3,b) * NC * hb) / hb) + 1
      if (kk < 1 .or. kk > NC) cycle
      write(u, '(I3,3ES24.16)') mesh%coords(4,b), mesh%coords(1:2,b) * NC * hb, hb
      write(u, '(*(ES24.16))') phi(1:NC,1:NC,kk,b)
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
endprogram t04_amr
