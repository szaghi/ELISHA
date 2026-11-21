!< Tutorial 2: boundary faces. Periodic, Dirichlet and Neumann faces on a line (y and z null axes), each pair against
!< its continuum solution, and what each kind writes into the ghost cell.
program t02_faces
!run t02_faces t02_faces
use elisha
implicit none
integer(I4P), parameter :: NB=2, NG=1                 ! 2 blocks along x, one ghost layer
real(R_P),    parameter :: PI=acos(-1._R_P)
character(*), parameter :: NAMES(4)=['periodic/periodic  ', 'Dirichlet/Dirichlet', 'Neumann/Neumann    ', &
                                     'Dirichlet/Neumann  ']
type(elisha_comm_object)  :: comm
integer(I4P)              :: stat, pair, level
real(R_P)                 :: err(2)
character(13)             :: ghost_lo, ghost_hi
character(:), allocatable :: errmsg
character(32)             :: arg
logical                   :: dump

call get_command_argument(1, arg) ; dump = trim(arg) == 'dump' ! the figures' data (docs/scripts), not shown
call comm%init(0, stat, errmsg) ; call check
print '(A)', '  faces (x low/high)     error 32 cells   error 64 cells   order   ghost lo     ghost hi'
do pair=1, 4
   do level=1, 2
      call solve_line(pair, 16 * level, err(level), ghost_lo, ghost_hi)
   enddo
   print '(2X,A,2ES17.3,F8.2,2X,2A13)', NAMES(pair), err, log(err(1) / err(2)) / log(2._R_P), ghost_lo, ghost_hi
enddo
call comm%destroy(stat)
contains
   subroutine solve_line(pair, nc, err, ghost_lo, ghost_hi)
   !< Solve phi'' = f on [0, 1] for one pair of faces, 2 blocks of nc cells; the error against the continuum u and
   !< the relation the solver wrote between each boundary cell and its ghost.
   integer(I4P), intent(in)  :: pair      !< 1 periodic, 2 Dirichlet/Dirichlet, 3 Neumann/Neumann, 4 Dirichlet/Neumann.
   integer(I4P), intent(in)  :: nc        !< Cells per block.
   real(R_P),    intent(out) :: err       !< Max |phi - u| / max |u|.
   character(*), intent(out) :: ghost_lo  !< What the low face's ghost holds.
   character(*), intent(out) :: ghost_hi  !< What the high face's ghost holds.
   type(elisha_mesh_object)   :: mesh
   type(elisha_solver_object) :: solver
   integer(I4P)               :: coords(4,NB), bc(2,3), b, i, n
   real(R_P), allocatable     :: f(:), phi(:,:,:,:), u(:)
   real(R_P)                  :: h, x, k, shift, c
   h = 1._R_P / (NB * nc)
   coords = reshape([0, 0, 0, 0, 1, 0, 0, 0], [4, NB])
   !region bc
   ! y and z are null axes (one cell, one block): their faces are periodic, so the problem is the line along x
   bc(:,2:3) = ELISHA_BC_PERIODIC
   select case(pair)
   case(1) ; bc(:,1) = [ELISHA_BC_PERIODIC,  ELISHA_BC_PERIODIC]  ; k = 2._R_P * PI ! u = sin(2 pi x + 1)
   case(2) ; bc(:,1) = [ELISHA_BC_DIRICHLET, ELISHA_BC_DIRICHLET] ; k = PI          ! u = sin(pi x)
   case(3) ; bc(:,1) = [ELISHA_BC_NEUMANN,   ELISHA_BC_NEUMANN]   ; k = PI          ! u = cos(pi x)
   case(4) ; bc(:,1) = [ELISHA_BC_DIRICHLET, ELISHA_BC_NEUMANN]   ; k = PI / 2      ! u = sin(pi x / 2)
   endselect
   call mesh%init(nijk=[nc, 1, 1], ngc=NG, nblk0=[NB, 1, 1], dx0=[h, 1._R_P, 1._R_P], emin=[0._R_P, 0._R_P, 0._R_P], &
                  periodic=[pair == 1, .true., .true.], null=[.false., .true., .true.], coords=coords, stat=stat, &
                  errmsg=errmsg)
   call check
   call solver%setup(mesh, comm, stat, errmsg, bc=bc) ! one level: the spectral path, an odd/even extension per kind
   call check
   !endregion bc
   ! f = u'' = -k^2 u, u the mode that meets the pair's faces (homogeneous values: 0 at a Dirichlet face, u' = 0 at
   ! a Neumann one)
   shift = merge(PI / 2, 0._R_P, pair == 3) + merge(1._R_P, 0._R_P, pair == 1) ! cos: the sine shifted by pi/2
   allocate(f(NB * nc), u(NB * nc), phi(1-NG:nc+NG,1-NG:1+NG,1-NG:1+NG,NB))
   n = 0
   do b=1, NB ; do i=1, nc
      n = n + 1 ; x = (mesh%coords(1,b) * nc + i - 0.5_R_P) * h
      u(n) = sin(k * x + shift) ; f(n) = -k**2 * u(n)
   enddo ; enddo
   call solver%solve_packed(mesh, f, 1._R_P, phi, stat, errmsg) ! periodic or Neumann/Neumann: f - mean f, singular
   call check
   !region error
   ! where no Dirichlet face fixes the constant, phi is defined up to one: compare with the means matched
   c = merge(0._R_P, (sum(phi(1:nc,1,1,:)) - sum(u)) / (NB * nc), pair == 2 .or. pair == 4)
   err = maxval(abs(pack(phi(1:nc,1,1,:), .true.) - c - u)) / maxval(abs(u))
   !endregion error
   !region ghosts
   ghost_lo = relation(phi(0,1,1,1), phi(1,1,1,1), phi(nc,1,1,NB))      ! inner: the cell next to it; far: the
   ghost_hi = relation(phi(nc+1,1,1,NB), phi(nc,1,1,NB), phi(1,1,1,1))  ! cell at the other end of the line
   !endregion ghosts
   if (dump .and. nc == 16) call dump_line(pair, mesh, phi, u)
   call solver%destroy
   call mesh%destroy
   endsubroutine solve_line

   function relation(ghost, inner, far) result(what)
   !< How a ghost value relates to the cells of the line: the far end's (periodic), its neighbour's (Neumann: a zero
   !< derivative at the face) or its neighbour's negated (Dirichlet: a zero value at the face).
   real(R_P), intent(in) :: ghost !< Ghost value.
   real(R_P), intent(in) :: inner !< The first interior value next to it.
   real(R_P), intent(in) :: far   !< The interior value at the other end of the line.
   character(13)         :: what  !< The relation.
   if (abs(ghost - far) <= 1.e-12_R_P * abs(far)) then
      what = ' = far end'
   elseif (abs(ghost - inner) <= 1.e-12_R_P * abs(inner)) then
      what = ' = inner'
   elseif (abs(ghost + inner) <= 1.e-12_R_P * abs(inner)) then
      what = ' = -inner'
   else
      what = ' ?'
   endif
   endfunction relation

   subroutine dump_line(pair, mesh, phi, u)
   !< Write x, phi (mean removed where it is defined up to a constant) and u, ghost cells included (docs/scripts).
   integer(I4P),             intent(in) :: pair !< Pair of faces.
   type(elisha_mesh_object), intent(in) :: mesh !< Mesh.
   real(R_P),                intent(in) :: phi(1-mesh%ngc:,1-mesh%ngc:,1-mesh%ngc:,:) !< Potential.
   real(R_P),                intent(in) :: u(:) !< Continuum solution at the cell centres.
   character(32) :: file
   integer(I4P)  :: v, b, i, nc
   real(R_P)     :: c
   nc = mesh%nijk(1)
   c = merge(0._R_P, (sum(phi(1:nc,1,1,:)) - sum(u)) / size(u), pair == 2 .or. pair == 4)
   write(file, '(A,I0,A)') 't02_faces-', pair, '.dat'
   open(newunit=v, file=trim(file), status='replace', action='write')
   do b=1, NB ; do i=merge(0, 1, b == 1), merge(nc + 1, nc, b == NB)
      write(v, '(2ES24.16)') (mesh%coords(1,b) * nc + i - 0.5_R_P) * mesh%dx0(1), phi(i,1,1,b) - c
   enddo ; enddo
   close(v)
   endsubroutine dump_line

   subroutine check
   !< Stop on a failed call, with the message.
   if (stat /= ELISHA_OK) then
      print '(A)', errmsg
      error stop 1
   endif
   endsubroutine check
endprogram t02_faces
