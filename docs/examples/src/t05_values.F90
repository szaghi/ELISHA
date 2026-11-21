!< Tutorial 5: boundary values. A box with a heated lid: Laplace's equation with Dirichlet values on every face, read
!< from the harmonic u = sin(pi x) sin(pi y) sinh(sqrt(2) pi z) / sinh(sqrt(2) pi), on a mesh refined towards the lid.
program t05_values
!run t05_values t05_values
use elisha
implicit none
integer(I4P), parameter :: NB=2, NG=1             ! 2^3 level-0 blocks on [0, 1]^3
real(R_P),    parameter :: PI=acos(-1._R_P), S2=sqrt(2._R_P) * PI
type(elisha_comm_object)  :: comm
integer(I4P)              :: stat, level, iters(2)
real(R_P)                 :: err(2)
character(:), allocatable :: errmsg
character(32)             :: arg
logical                   :: dump

call get_command_argument(1, arg) ; dump = trim(arg) == 'dump' ! the figures' data (docs/scripts), not shown
call comm%init(0, stat, errmsg) ; call check
do level=1, 2
   call solve_lid(4 * 2**level, err(level), iters(level))
enddo
print '(A)', '  cells/block   FCG   max |phi - u| / max |u|'
print '(I8,A,I9,ES17.3)', 8, '^3', iters(1), err(1)
print '(I8,A,I9,ES17.3)', 16, '^3', iters(2), err(2)
print '(A,F5.2)', '  order ', log(err(1) / err(2)) / log(2._R_P)
call comm%destroy(stat)
contains
   subroutine solve_lid(nc, err, iters)
   !< Solve the lid problem with blocks of nc^3 cells; return the error and the FCG iterations.
   integer(I4P), intent(in)  :: nc    !< Cells per block per axis.
   real(R_P),    intent(out) :: err   !< Max |phi - u| / max |u|.
   integer(I4P), intent(out) :: iters !< FCG iterations.
   type(elisha_mesh_object)        :: mesh
   type(elisha_solver_object)      :: solver
   type(elisha_diagnostics_object) :: diag
   integer(I4P)                    :: coords(4,NB**3+4*7), bc(2,3), b, i, j, k, o
   integer(I4P), allocatable       :: kinds(:), dir(:)
   real(R_P),    allocatable       :: f(:), phi(:,:,:,:), xf(:,:), g(:)
   real(R_P)                       :: h, x(3)
   h = 1._R_P / (NB * nc)
   ! the upper level-0 blocks (k = 1) refined to level 1: the lid's gradients are resolved twice as finely
   b = 0
   do k=0, NB-1 ; do j=0, NB-1 ; do i=0, NB-1
      if (k == 1) then
         do o=0, 7
            b = b + 1 ; coords(:,b) = [2*i + ibits(o, 0, 1), 2*j + ibits(o, 1, 1), 2*k + ibits(o, 2, 1), 1]
         enddo
      else
         b = b + 1 ; coords(:,b) = [i, j, k, 0]
      endif
   enddo ; enddo ; enddo
   call mesh%init(nijk=[nc, nc, nc], ngc=NG, nblk0=[NB, NB, NB], dx0=[h, h, h], emin=[0._R_P, 0._R_P, 0._R_P], &
                  periodic=[.false., .false., .false.], null=[.false., .false., .false.], coords=coords(:,1:b), &
                  stat=stat, errmsg=errmsg)
   call check
   !region setup
   ! Dirichlet on every face: boundary values need the hierarchical path (here chosen by the two levels anyway)
   bc = ELISHA_BC_DIRICHLET
   call solver%setup(mesh, comm, stat, errmsg, bc=bc, tolerance=1.e-12_R_P)
   call check
   !endregion setup
   !region values
   ! this rank's boundary faces: the kind of each, its outward axis (-3 to 3) and where its value lives (here the
   ! face centres); the values go to solve_packed in the same order
   call solver%boundary_faces(kinds, dir, xf, stat, errmsg)
   call check
   allocate(g(size(kinds)))
   do i=1, size(kinds)
      g(i) = lid(xf(:,i))
   enddo
   !endregion values
   !region solve
   allocate(f(mesh%nb * nc**3), phi(1-NG:nc+NG,1-NG:nc+NG,1-NG:nc+NG,mesh%nb))
   f = 0._R_P                                                    ! Laplace: no source
   call solver%solve_packed(mesh, f, 1._R_P, phi, stat, errmsg, ghost=g)
   call check
   diag = solver%diagnostics() ; iters = diag%iterations
   !endregion solve
   err = 0._R_P
   do b=1, mesh%nb ; do k=1, nc ; do j=1, nc ; do i=1, nc
      x = (mesh%coords(1:3,b) * nc + [i, j, k] - 0.5_R_P) * h / 2**mesh%coords(4,b)
      err = max(err, abs(phi(i,j,k,b) - lid(x)))
   enddo ; enddo ; enddo ; enddo
   if (dump .and. nc == 8) call dump_slice('t05_values-slice.dat', mesh, phi)
   call solver%destroy
   call mesh%destroy
   endsubroutine solve_lid

   pure function lid(x) result(u)
   !< The harmonic u: zero on five faces, sin(pi x) sin(pi y) on the lid z = 1.
   real(R_P), intent(in) :: x(3) !< Point.
   real(R_P)             :: u    !< Value.
   u = sin(PI * x(1)) * sin(PI * x(2)) * sinh(S2 * x(3)) / sinh(S2)
   endfunction lid

   subroutine dump_slice(file, mesh, phi)
   !< The plane y = 1/2 (just below it) as x-z blocks (docs/scripts): per block crossing it, its level, its lower
   !< corner (x, z) and spacing, then its nc x nc values (i fastest, then k).
   character(*),             intent(in) :: file !< Output file.
   type(elisha_mesh_object), intent(in) :: mesh !< Mesh.
   real(R_P),                intent(in) :: phi(1-mesh%ngc:,1-mesh%ngc:,1-mesh%ngc:,:) !< Potential.
   integer(I4P) :: u, b, jj
   real(R_P)    :: hb
   open(newunit=u, file=file, status='replace', action='write')
   do b=1, mesh%nb
      hb = mesh%dx0(2) / 2**mesh%coords(4,b)
      jj = floor((0.5_R_P - 0.25_R_P * mesh%dx0(2) - mesh%coords(2,b) * mesh%nijk(2) * hb) / hb) + 1
      if (jj < 1 .or. jj > mesh%nijk(2)) cycle
      write(u, '(I3,3ES24.16)') mesh%coords(4,b), mesh%coords(1,b) * mesh%nijk(1) * hb, &
                                mesh%coords(3,b) * mesh%nijk(3) * hb, hb
      write(u, '(*(ES24.16))') phi(1:mesh%nijk(1),jj,1:mesh%nijk(3),b)
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
endprogram t05_values
