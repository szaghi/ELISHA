!< ELISHA, the setup checks: ADAM's refusals of a solver's arguments, as pure procedures that return a status.
module elisha_setup_checks
!< ELISHA, the setup checks: ADAM's refusals of a solver's arguments, as pure procedures that return a status.
!<
!< FLAIL refused a setup with `error stop` scattered over its facade, its spectral solver and its multigrid (ADAM
!< 669d0bdd). Here each family of refusals is one pure check, returning `ELISHA_OK` or the code of the first failure,
!< and the solvers compose them (M3, M4). The mesh's own refusals (levels, coverage, balance, key limit) are
!< `elisha_mesh_object%validate`'s. Two deliberate differences from ADAM, both stricter:
!< - an unknown kind is reported before the pair rules of its axis (ADAM checked it last);
!< - a lone periodic face is refused on every path: ADAM's multigrid accepted `[periodic, Dirichlet]` on a
!<   non-periodic axis (its periodicity check compares the grid with "both faces periodic", which both deny).
!<
!< Checks, their codes (ELISHA_ERR_*) and where ADAM refused the same:
!< - `elisha_check_bc`: BC_UNKNOWN, BC_PERIODIC_PAIR, BC_GHOST_PAIR, BC_ZERO_GHOST_PAIR, BC_FREE_GHOSTS (fft
!<   initialize_blocks, mg initialize);
!< - `elisha_check_dg_order`: DG_ORDER (fft initialize_blocks);
!< - `elisha_check_isolated`: ISOLATED_BC, ISOLATED_WIDE, ISOLATED_DIMENSION, ISOLATED_PERIODIC, ISOLATED_CELLS (facade
!<   setup, fft initialize and initialize_blocks, mg initialize);
!< - `elisha_check_periodicity`: BC_PERIODICITY; `elisha_check_unbounded`: BC_MISSING (fft and mg initialize);
!< - `elisha_check_reach`: DG_REACH (fft initialize_blocks);
!< - `elisha_check_path`: DG_AMR, BC_FREE_GHOSTS, REDUNDANT, GHOST_VALUES (facade setup and solve, fft, mg);
!< - `elisha_check_ghost_count`: GHOST_COUNT (mg solve);
!< - `elisha_check_isolated_mesh`, collective: ISOLATED_BOUNDARY_LEAF (mg initialize).
use penf, only : I4P
use elisha_kind_parameters, only : R_P
use elisha_comm, only : elisha_comm_object
use elisha_mesh, only : elisha_mesh_object
use elisha_parameters, only : ELISHA_BC_PERIODIC, ELISHA_BC_DIRICHLET_GHOST, ELISHA_BC_ZERO_GHOSTS, &
                              ELISHA_BC_FREE_GHOSTS, ELISHA_BC_KINDS, ELISHA_PATH_SPECTRAL, ELISHA_PATH_HIERARCHICAL
use elisha_status
implicit none
private
public :: elisha_check_bc, elisha_check_dg_order, elisha_check_ghost_count, elisha_check_isolated
public :: elisha_check_isolated_mesh, elisha_check_path, elisha_check_periodicity, elisha_check_reach
public :: elisha_check_unbounded

real(R_P), parameter :: CUBIC_TOL=1.e-12_R_P !< Relative tolerance of cubic cells (ADAM's).

contains
   pure subroutine elisha_check_bc(bc, null, axis_order, stat, errmsg)
   !< The kinds of every face: known; per active axis, both faces periodic or neither, both ghost-Dirichlet or neither,
   !< both zero/free ghosts or neither; free ghosts only with the wide operator on that axis (`axis_order` /= 0).
   integer(I4P),              intent(in)            :: bc(2,3)       !< Kinds [lo/hi, axis].
   logical,                   intent(in)            :: null(3)       !< Null axes (not checked).
   integer(I4P),              intent(in)            :: axis_order(3) !< Operator order per axis (0 compact).
   integer(I4P),              intent(out)           :: stat          !< Status.
   character(:), allocatable, intent(out), optional :: errmsg        !< Message.
   integer(I4P)                                     :: d, k          !< Counters.

   stat = ELISHA_OK
   do d=1, 3
      if (null(d)) cycle
      do k=1, 2
         if (all(bc(k,d) /= ELISHA_BC_KINDS)) then
            call elisha_set_error(stat, errmsg, ELISHA_ERR_BC_UNKNOWN, 'kind '//itoa(bc(k,d))//' on axis '//itoa(d))
            return
         endif
      enddo
      if (count(bc(:,d) == ELISHA_BC_PERIODIC) == 1) then
         call elisha_set_error(stat, errmsg, ELISHA_ERR_BC_PERIODIC_PAIR, 'axis '//itoa(d))
         return
      endif
      if (count(bc(:,d) == ELISHA_BC_DIRICHLET_GHOST) == 1) then
         call elisha_set_error(stat, errmsg, ELISHA_ERR_BC_GHOST_PAIR, 'axis '//itoa(d))
         return
      endif
      if (count(bc(:,d) == ELISHA_BC_ZERO_GHOSTS .or. bc(:,d) == ELISHA_BC_FREE_GHOSTS) == 1) then
         call elisha_set_error(stat, errmsg, ELISHA_ERR_BC_ZERO_GHOST_PAIR, 'axis '//itoa(d)// &
                               ' (a reflection at the other end couples the parity sub-lattices)')
         return
      endif
      if (any(bc(:,d) == ELISHA_BC_FREE_GHOSTS) .and. axis_order(d) == 0_I4P) then
         call elisha_set_error(stat, errmsg, ELISHA_ERR_BC_FREE_GHOSTS, 'axis '//itoa(d)//' with the compact operator')
         return
      endif
   enddo
   endsubroutine elisha_check_bc

   pure subroutine elisha_check_dg_order(dg_order, stat, errmsg)
   !< The wide operator's order: 0 (compact), 2, 4, 6 or 8.
   integer(I4P),              intent(in)            :: dg_order !< Order.
   integer(I4P),              intent(out)           :: stat     !< Status.
   character(:), allocatable, intent(out), optional :: errmsg   !< Message.

   stat = ELISHA_OK
   if (all(dg_order /= [0_I4P, 2_I4P, 4_I4P, 6_I4P, 8_I4P])) &
      call elisha_set_error(stat, errmsg, ELISHA_ERR_DG_ORDER, 'order '//itoa(dg_order))
   endsubroutine elisha_check_dg_order

   pure subroutine elisha_check_isolated(isolated, has_bc, dg_order, null, periodic, dx, stat, errmsg)
   !< Isolated boundaries: no boundary kinds, the compact operator, 3-D, no periodic axis, cubic cells.
   logical,                   intent(in)            :: isolated    !< Isolated boundaries asked.
   logical,                   intent(in)            :: has_bc      !< Boundary kinds (or ghost-Dirichlet) given.
   integer(I4P),              intent(in)            :: dg_order    !< Operator order (0 compact).
   logical,                   intent(in)            :: null(3)     !< Null axes.
   logical,                   intent(in)            :: periodic(3) !< Periodic axes.
   real(R_P),                 intent(in)            :: dx(3)       !< Cell spacing.
   integer(I4P),              intent(out)           :: stat        !< Status.
   character(:), allocatable, intent(out), optional :: errmsg      !< Message.
   integer(I4P)                                     :: d           !< Counter.

   stat = ELISHA_OK
   if (.not.isolated) return
   if (has_bc) then
      call elisha_set_error(stat, errmsg, ELISHA_ERR_ISOLATED_BC, 'the ghost values are computed (the James step)')
      return
   endif
   if (dg_order /= 0_I4P) then
      call elisha_set_error(stat, errmsg, ELISHA_ERR_ISOLATED_WIDE, 'order '//itoa(dg_order)// &
                            ' (the lattice Green''s function is the compact operator''s)')
      return
   endif
   do d=1, 3
      if (null(d)) then
         call elisha_set_error(stat, errmsg, ELISHA_ERR_ISOLATED_DIMENSION, 'axis '//itoa(d)//' is null')
         return
      endif
      if (periodic(d)) then
         call elisha_set_error(stat, errmsg, ELISHA_ERR_ISOLATED_PERIODIC, 'axis '//itoa(d)//' is periodic')
         return
      endif
   enddo
   if (abs(dx(2) - dx(1)) > CUBIC_TOL * dx(1) .or. abs(dx(3) - dx(1)) > CUBIC_TOL * dx(1)) &
      call elisha_set_error(stat, errmsg, ELISHA_ERR_ISOLATED_CELLS, 'the lattice Green''s function is the unit '// &
                            'cubic lattice''s')
   endsubroutine elisha_check_isolated

   pure subroutine elisha_check_periodicity(bc, periodic, null, stat, errmsg)
   !< With boundary kinds: an active axis is periodic in the mesh exactly when both its faces are periodic in `bc`.
   integer(I4P),              intent(in)            :: bc(2,3)     !< Kinds [lo/hi, axis].
   logical,                   intent(in)            :: periodic(3) !< The mesh's periodic axes.
   logical,                   intent(in)            :: null(3)     !< Null axes.
   integer(I4P),              intent(out)           :: stat        !< Status.
   character(:), allocatable, intent(out), optional :: errmsg      !< Message.
   integer(I4P)                                     :: d           !< Counter.

   stat = ELISHA_OK
   do d=1, 3
      if (null(d)) cycle
      if (periodic(d) .neqv. all(bc(:,d) == ELISHA_BC_PERIODIC)) then
         call elisha_set_error(stat, errmsg, ELISHA_ERR_BC_PERIODICITY, 'axis '//itoa(d)//' is '// &
                               merge('    periodic', 'not periodic', periodic(d))//' in the mesh and '// &
                               merge('    periodic', 'not periodic', all(bc(:,d) == ELISHA_BC_PERIODIC))//' in bc')
         return
      endif
   enddo
   endsubroutine elisha_check_periodicity

   pure subroutine elisha_check_unbounded(path, periodic, null, stat, errmsg)
   !< Without boundary kinds (and not isolated): the spectral path needs every active axis periodic; the hierarchical
   !< one every active axis periodic or none (the bounded ones ghost-Dirichlet; a mixed mesh would get a silent zero).
   integer(I4P),              intent(in)            :: path        !< ELISHA_PATH_SPECTRAL or _HIERARCHICAL.
   logical,                   intent(in)            :: periodic(3) !< The mesh's periodic axes.
   logical,                   intent(in)            :: null(3)     !< Null axes.
   integer(I4P),              intent(out)           :: stat        !< Status.
   character(:), allocatable, intent(out), optional :: errmsg      !< Message.
   integer(I4P)                                     :: d           !< Counter.

   stat = ELISHA_OK
   if (path == ELISHA_PATH_SPECTRAL) then
      do d=1, 3
         if (.not.null(d) .and. .not.periodic(d)) then
            call elisha_set_error(stat, errmsg, ELISHA_ERR_BC_MISSING, 'axis '//itoa(d)//' is not periodic: give '// &
                                  'the kind of every face (or isolated boundaries, 3-D)')
            return
         endif
      enddo
   elseif (any(periodic .and. .not.null) .and. .not.all(periodic .or. null)) then
      call elisha_set_error(stat, errmsg, ELISHA_ERR_BC_MISSING, 'periodic and non-periodic axes: give the kind of '// &
                            'every face')
   endif
   endsubroutine elisha_check_unbounded

   pure subroutine elisha_check_reach(n, null, dg_order, pcg, stat, errmsg)
   !< On the CG path of the wide operator (zero/free-ghost faces) every active axis has at least `dg_order` cells.
   integer(I4P),              intent(in)            :: n(3)     !< Cells per axis.
   logical,                   intent(in)            :: null(3)  !< Null axes.
   integer(I4P),              intent(in)            :: dg_order !< Operator order.
   logical,                   intent(in)            :: pcg      !< The CG path.
   integer(I4P),              intent(out)           :: stat     !< Status.
   character(:), allocatable, intent(out), optional :: errmsg   !< Message.
   integer(I4P)                                     :: d        !< Counter.

   stat = ELISHA_OK
   if (.not.pcg) return
   do d=1, 3
      if (null(d)) cycle
      if (n(d) < dg_order) then
         call elisha_set_error(stat, errmsg, ELISHA_ERR_DG_REACH, itoa(n(d))//' cells along axis '//itoa(d)// &
                               ', the reach is '//itoa(dg_order))
         return
      endif
   enddo
   endsubroutine elisha_check_reach

   pure subroutine elisha_check_path(path, uniform, dg_order, bc, isolated, redundant, pcg, has_ghost, stat, errmsg)
   !< What the solve path admits: the wide operator and free ghosts only on the spectral path of a uniform mesh; the
   !< redundant spectral solve only compact and not isolated; boundary values only hierarchical and not isolated.
   integer(I4P),              intent(in)            :: path      !< ELISHA_PATH_SPECTRAL or _HIERARCHICAL.
   logical,                   intent(in)            :: uniform   !< The mesh is at one level.
   integer(I4P),              intent(in)            :: dg_order  !< Operator order.
   integer(I4P),              intent(in)            :: bc(2,3)   !< Kinds.
   logical,                   intent(in)            :: isolated  !< Isolated boundaries.
   logical,                   intent(in)            :: redundant !< The redundant spectral solve (every rank).
   logical,                   intent(in)            :: pcg       !< The CG path (zero/free ghosts with D(G)).
   logical,                   intent(in)            :: has_ghost !< Boundary values given.
   integer(I4P),              intent(out)           :: stat      !< Status.
   character(:), allocatable, intent(out), optional :: errmsg    !< Message.

   stat = ELISHA_OK
   if (dg_order /= 0_I4P .and. (path == ELISHA_PATH_HIERARCHICAL .or. .not.uniform)) then
      call elisha_set_error(stat, errmsg, ELISHA_ERR_DG_AMR, 'a composite wide operator across 2:1 faces is not '// &
                            'defined (ADAM #84 E7)')
   elseif (path == ELISHA_PATH_HIERARCHICAL .and. any(bc == ELISHA_BC_FREE_GHOSTS)) then
      call elisha_set_error(stat, errmsg, ELISHA_ERR_BC_FREE_GHOSTS, 'free-space ghosts on the hierarchical path')
   elseif (redundant .and. (isolated .or. pcg)) then
      call elisha_set_error(stat, errmsg, ELISHA_ERR_REDUNDANT, 'isolated or CG path')
   elseif (has_ghost .and. isolated) then
      call elisha_set_error(stat, errmsg, ELISHA_ERR_GHOST_VALUES, 'isolated boundaries compute them')
   elseif (has_ghost .and. path == ELISHA_PATH_SPECTRAL) then
      call elisha_set_error(stat, errmsg, ELISHA_ERR_GHOST_VALUES, 'the spectral path solves the homogeneous problem')
   endif
   endsubroutine elisha_check_path

   pure subroutine elisha_check_ghost_count(nghost, nbnd, stat, errmsg)
   !< At least one boundary value per boundary face of this rank.
   integer(I4P),              intent(in)            :: nghost !< Values given.
   integer(I4P),              intent(in)            :: nbnd   !< Boundary faces.
   integer(I4P),              intent(out)           :: stat   !< Status.
   character(:), allocatable, intent(out), optional :: errmsg !< Message.

   stat = ELISHA_OK
   if (nghost < nbnd) call elisha_set_error(stat, errmsg, ELISHA_ERR_GHOST_COUNT, itoa(nghost)//' values for '// &
                                            itoa(nbnd)//' faces')
   endsubroutine elisha_check_ghost_count

   subroutine elisha_check_isolated_mesh(mesh, comm, stat, errmsg)
   !< Isolated boundaries on the hierarchical path: every leaf on a domain face is at the coarsest level (the ghost
   !< lattice of the James step is the coarsest one). Collective: every rank gets the same status.
   type(elisha_mesh_object),  intent(in)            :: mesh   !< Mesh (validated).
   type(elisha_comm_object),  intent(in)            :: comm   !< Communication context.
   integer(I4P),              intent(out)           :: stat   !< Status.
   character(:), allocatable, intent(out), optional :: errmsg !< Message.
   integer(I4P)                                     :: b, d, c(1), st !< Counters, agreed code, status.
   character(:), allocatable                        :: mine   !< This rank's message.

   c = ELISHA_OK ; mine = ''
   do b=1, mesh%nb
      if (mesh%coords(4,b) == mesh%lmin) cycle
      do d=1, 3
         if (mesh%null(d)) cycle
         if (mesh%coords(d,b) == 0 .or. int(mesh%coords(d,b) + 1, kind(mesh%nblk0)) == &
             mesh%blocks_along(mesh%coords(4,b), d)) then
            c = ELISHA_ERR_ISOLATED_BOUNDARY_LEAF
            mine = 'a leaf at level '//itoa(mesh%coords(4,b))//' touches the domain boundary (coarsest '// &
                   itoa(mesh%lmin)//')'
            exit
         endif
      enddo
      if (c(1) /= ELISHA_OK) exit
   enddo
   b = c(1)
   call comm%max_i4(c, st)
   if (st /= ELISHA_OK) then
      call elisha_set_error(stat, errmsg, st, 'elisha_check_isolated_mesh: a collective failed')
   elseif (c(1) /= ELISHA_OK) then
      if (b == c(1)) then
         call elisha_set_error(stat, errmsg, c(1), mine)
      else
         call elisha_set_error(stat, errmsg, c(1), 'on another rank')
      endif
   else
      stat = ELISHA_OK
   endif
   endsubroutine elisha_check_isolated_mesh

   pure function itoa(i) result(s)
   !< An integer as text.
   integer(I4P), intent(in)  :: i !< Integer.
   character(:), allocatable :: s !< Text.
   character(16)             :: b !< Buffer.

   write(b, '(I0)') i
   s = trim(b)
   endfunction itoa
endmodule elisha_setup_checks
