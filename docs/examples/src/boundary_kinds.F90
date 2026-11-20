!< Boundary kinds: Dirichlet and Neumann faces on the spectral path; boundary values on the hierarchical path.
program boundary_kinds
!run boundary_kinds boundary_kinds
use elisha
implicit none
integer(I4P), parameter :: NB=2, NC=8, NG=1      ! 2^3 level-0 blocks of 8^3 cells on [0, 1]^3
real(R_P),    parameter :: H=1._R_P / (NB * NC)   ! cell spacing at level 0
real(R_P),    parameter :: PI=acos(-1._R_P)
type(elisha_comm_object)   :: comm
type(elisha_mesh_object)   :: mesh
type(elisha_solver_object) :: solver
integer(I4P)              :: coords(4,NB**3+7), bc(2,3), stat, b, i, j, k, n, o
integer(I4P), allocatable :: kinds(:), dir(:)
real(R_P),    allocatable :: f(:), phi(:,:,:,:), xf(:,:), g(:)
real(R_P)                 :: x(3), d_lo, n_hi, err
character(:), allocatable :: errmsg
call comm%init(0, stat, errmsg) ; call check
!region spectral
! x: Dirichlet at the low face, Neumann at the high one; y: the reverse; z: Dirichlet at both (homogeneous values)
bc(:,1) = [ELISHA_BC_DIRICHLET, ELISHA_BC_NEUMANN]
bc(:,2) = [ELISHA_BC_NEUMANN, ELISHA_BC_DIRICHLET]
bc(:,3) = [ELISHA_BC_DIRICHLET, ELISHA_BC_DIRICHLET]
call make_mesh(refined=.false.)
call solver%setup(mesh, comm, stat, errmsg, bc=bc) ! one level: the spectral path (odd/even extensions)
call check
call solver%solve_packed(mesh, f, 1._R_P, phi, stat, errmsg)
call check
! the ghost layer carries the kinds: phi_ghost = -phi_boundary at a Dirichlet face, = phi_boundary at a Neumann one
d_lo = 0._R_P ; n_hi = 0._R_P
do b=1, mesh%nb
   if (mesh%coords(1,b) == 0)      d_lo = max(d_lo, maxval(abs(phi(0,1:NC,1:NC,b) + phi(1,1:NC,1:NC,b))))
   if (mesh%coords(1,b) == NB - 1) n_hi = max(n_hi, maxval(abs(phi(NC+1,1:NC,1:NC,b) - phi(NC,1:NC,1:NC,b))))
enddo
print '(A,L1,A,L1)', 'spectral ', solver%path() == ELISHA_PATH_SPECTRAL, ', x low Dirichlet ', &
                     d_lo <= 1.e-12_R_P * maxval(abs(phi))
print '(A,L1)', 'x high Neumann ', n_hi <= 1.e-12_R_P * maxval(abs(phi))
!endregion spectral
!region values
! a regrid to a 2:1 mesh, Dirichlet everywhere, no source and the values of u = 1 + x + 2 y - z on the faces: the
! compact operator is exact for linear u, across the 2:1 seam too, so phi = u at every cell centre
call make_mesh(refined=.true.)
bc = ELISHA_BC_DIRICHLET
call solver%setup(mesh, comm, stat, errmsg, bc=bc, tolerance=1.e-12_R_P) ! two levels: the hierarchical path
call check
call solver%boundary_faces(kinds, dir, xf, stat, errmsg) ! this rank's faces: kind, outward axis, where the value is
call check
g = 1._R_P + xf(1,:) + 2._R_P * xf(2,:) - xf(3,:)
f = 0._R_P
call solver%solve_packed(mesh, f, 1._R_P, phi, stat, errmsg, ghost=g)
call check
err = 0._R_P
do b=1, mesh%nb ; do k=1, NC ; do j=1, NC ; do i=1, NC
   x = (mesh%coords(1:3,b) * NC + [i, j, k] - 0.5_R_P) * H / 2**mesh%coords(4,b)
   err = max(err, abs(phi(i,j,k,b) - (1._R_P + x(1) + 2._R_P * x(2) - x(3))))
enddo ; enddo ; enddo ; enddo
print '(A,L1,A,I0,A,L1)', 'hierarchical ', solver%is_hierarchical(), ', boundary faces ', size(g), &
                          ', phi = u to 1e-9 ', err <= 1.e-9_R_P
!endregion values
call solver%destroy
call mesh%destroy
call comm%destroy(stat)
contains
   subroutine make_mesh(refined)
   !< The mesh on [0, 1]^3, non-periodic: 2^3 level-0 blocks, block (0,0,0) refined to level 1 when `refined`; the
   !< source sin(pi x) sin(pi y) sin(pi z) and the potential sized for it.
   logical, intent(in) :: refined !< Refine block (0,0,0).
   b = 0
   do k=0, NB-1 ; do j=0, NB-1 ; do i=0, NB-1
      if (refined .and. all([i, j, k] == 0)) then
         do o=0, 7
            b = b + 1 ; coords(:,b) = [ibits(o, 0, 1), ibits(o, 1, 1), ibits(o, 2, 1), 1]
         enddo
      else
         b = b + 1 ; coords(:,b) = [i, j, k, 0]
      endif
   enddo ; enddo ; enddo
   call mesh%init(nijk=[NC, NC, NC], ngc=NG, nblk0=[NB, NB, NB], dx0=[H, H, H], emin=[0._R_P, 0._R_P, 0._R_P], &
                  periodic=[.false., .false., .false.], null=[.false., .false., .false.], coords=coords(:,1:b), &
                  stat=stat, errmsg=errmsg)
   call check
   if (allocated(f)) deallocate(f, phi)
   allocate(f(mesh%nb * NC**3), phi(1-NG:NC+NG,1-NG:NC+NG,1-NG:NC+NG,mesh%nb))
   n = 0
   do b=1, mesh%nb ; do k=1, NC ; do j=1, NC ; do i=1, NC
      n = n + 1 ; x = (mesh%coords(1:3,b) * NC + [i, j, k] - 0.5_R_P) * H / 2**mesh%coords(4,b)
      f(n) = product(sin(PI * x))
   enddo ; enddo ; enddo ; enddo
   endsubroutine make_mesh

   subroutine check
   !< Stop on a failed call, with the message.
   if (stat /= ELISHA_OK) then
      print '(A)', errmsg
      error stop 1
   endif
   endsubroutine check
endprogram boundary_kinds
