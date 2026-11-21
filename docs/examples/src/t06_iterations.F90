!< Tutorial 6: watching the FCG converge. Two embers on a periodic 2-D AMR mesh; the hierarchical solve stopped after
!< k iterations, k = 1, 2, ..., returns ELISHA_ERR_NOT_CONVERGED with its k-th iterate, until it converges; the
!< residual history of the symmetric V-cycle (the default) and of ADAM's.
program t06_iterations
!run t06_iterations t06_iterations
use elisha
implicit none
integer(I4P), parameter :: NB=4, NC=8, NG=1, LMAX=2, KMAX=30 ! 4^2 level-0 blocks of 8^2 cells, 2 levels of refinement
real(R_P),    parameter :: H=1._R_P / (NB * NC), SIGMA=0.035_R_P
real(R_P),    parameter :: EMBER(2,2)=reshape([0.38_R_P, 0.42_R_P, 0.62_R_P, 0.57_R_P], [2, 2]) ! the two centres
type(elisha_comm_object)        :: comm
type(elisha_mesh_object)        :: mesh
type(elisha_solver_object)      :: solver
type(elisha_diagnostics_object) :: diag
integer(I4P), allocatable :: coords(:,:)
real(R_P),    allocatable :: f(:), phi(:,:,:,:), frames(:,:,:,:,:)
integer(I4P)              :: stat, nleaf, b, i, j, k, n, cycle_kind, iters(2)
real(R_P)                 :: x(2), hb, history(KMAX,2)
character(:), allocatable :: errmsg
character(32)             :: arg
logical                   :: dump

call get_command_argument(1, arg) ; dump = trim(arg) == 'dump' ! the figures' data (docs/scripts), not shown
call comm%init(0, stat, errmsg) ; call check
allocate(coords(4,NB**2 * 4**LMAX)) ; nleaf = 0
do j=0, NB-1 ; do i=0, NB-1
   call refine([i, j, 0, 0])
enddo ; enddo
!region mesh
! z is a null axis: one cell, one block; the octree is a quadtree
call mesh%init(nijk=[NC, NC, 1], ngc=NG, nblk0=[NB, NB, 1], dx0=[H, H, 1._R_P], emin=[0._R_P, 0._R_P, 0._R_P], &
               periodic=[.true., .true., .true.], null=[.false., .false., .true.], coords=coords(:,1:nleaf), &
               stat=stat, errmsg=errmsg)
call check
!endregion mesh
allocate(f(mesh%nb * NC**2), phi(1-NG:NC+NG,1-NG:NC+NG,1-NG:1+NG,mesh%nb))
n = 0
do b=1, mesh%nb
   hb = H / 2**mesh%coords(4,b)
   do j=1, NC ; do i=1, NC
      n = n + 1 ; x = (mesh%coords(1:2,b) * NC + [i, j] - 0.5_R_P) * hb
      f(n) = exp(-sum((x - EMBER(:,1))**2) / (2 * SIGMA**2)) + exp(-sum((x - EMBER(:,2))**2) / (2 * SIGMA**2))
   enddo ; enddo
enddo
if (dump) allocate(frames(1:NC,1:NC,mesh%nb,0:KMAX,2))
!region history
! a fresh setup per k (a solve warm-starts from the previous one of its setup): each solve starts from phi = 0
history = 0._R_P ; iters = 0
do cycle_kind=1, 2
   do k=1, KMAX
      call solver%setup(mesh, comm, stat, errmsg, max_iterations=k, tolerance=1.e-10_R_P, &
                        vcycle=merge(ELISHA_VCYCLE_SYMMETRIC, ELISHA_VCYCLE_ADAM, cycle_kind == 1))
      call check
      call solver%solve_packed(mesh, f, 1._R_P, phi, stat, errmsg)
      diag = solver%diagnostics()
      history(k,cycle_kind) = diag%residual
      if (dump) frames(:,:,:,k,cycle_kind) = phi(1:NC,1:NC,1,:)
      if (stat == ELISHA_OK) then                  ! converged within k iterations
         iters(cycle_kind) = diag%iterations
         exit
      elseif (stat /= ELISHA_ERR_NOT_CONVERGED) then ! a real failure
         call check
      endif                                        ! not converged: phi is the k-th iterate, its ghosts filled
   enddo
enddo
!endregion history
print '(A)', '   k   residual, symmetric   residual, ADAM''s'
do k=1, maxval(iters)
   print '(I4,2A21)', k, fmt_residual(history(k,1), k <= iters(1)), fmt_residual(history(k,2), k <= iters(2))
enddo
print '(A,I0,A,I0,A)', 'converged in ', iters(1), ' (symmetric) and ', iters(2), ' (ADAM''s) iterations'
if (dump) call write_frames
call solver%destroy
call mesh%destroy
call comm%destroy(stat)
contains
   recursive subroutine refine(c4)
   !< Emit the block c4 as a leaf, or its 4 children when it is above level LMAX and an ember is within the reach of
   !< its level: 0.22 at level 0, 0.08 at level 1 (wide enough for the children of the level-1 blocks to be 2:1).
   integer(I4P), intent(in) :: c4(4) !< Block (i, j, 0, level).
   integer(I4P) :: o
   real(R_P)    :: lo(2), hi(2), reach
   lo = c4(1:2) * NC * H / 2**c4(4) ; hi = lo + NC * H / 2**c4(4)
   reach = merge(0.22_R_P, 0.08_R_P, c4(4) == 0)
   if (c4(4) < LMAX .and. (norm2(max(lo - EMBER(:,1), 0._R_P, EMBER(:,1) - hi)) < reach .or. &
                           norm2(max(lo - EMBER(:,2), 0._R_P, EMBER(:,2) - hi)) < reach)) then
      do o=0, 3
         call refine([2*c4(1) + ibits(o, 0, 1), 2*c4(2) + ibits(o, 1, 1), 0, c4(4) + 1])
      enddo
   else
      nleaf = nleaf + 1 ; coords(:,nleaf) = c4
   endif
   endsubroutine refine

   function fmt_residual(r, valid) result(s)
   !< The residual with two digits, blank after the solve has converged.
   real(R_P), intent(in) :: r     !< Residual.
   logical,   intent(in) :: valid !< The solve ran this iteration.
   character(21)         :: s     !< Formatted.
   s = ''
   if (valid) write(s, '(ES21.1)') r
   endfunction fmt_residual

   subroutine write_frames
   !< The mesh and the iterates of both V-cycles (docs/scripts): a header line per block (level, lower corner, spacing),
   !< then per V-cycle and k the values of every block, the converged solution last.
   integer(I4P) :: u, c
   open(newunit=u, file='t06_iterations-frames.dat', status='replace', action='write')
   write(u, '(4I6)') mesh%nb, NC, iters
   do b=1, mesh%nb
      hb = H / 2**mesh%coords(4,b)
      write(u, '(I3,3ES24.16)') mesh%coords(4,b), mesh%coords(1:2,b) * NC * hb, hb
   enddo
   do c=1, 2 ; do k=1, iters(c) ; do b=1, mesh%nb
      write(u, '(*(ES16.8))') frames(:,:,b,k,c)
   enddo ; enddo ; enddo
   close(u)
   endsubroutine write_frames

   subroutine check
   !< Stop on a failed call, with the message.
   if (stat /= ELISHA_OK) then
      print '(A)', errmsg
      error stop 1
   endif
   endsubroutine check
endprogram t06_iterations
