!< ELISHA test: the setup checks, one case per ADAM refusal they replace, and the cases they must accept.
program elisha_test_setup_checks_mpi
!< ELISHA test: the setup checks, one case per ADAM refusal they replace, and the cases they must accept.
!<
!< Each refusal of ADAM's FLAIL setup (ADAM 669d0bdd: the facade's `setup`/`solve`, `poisson_fft_object%initialize`
!< and `initialize_blocks`, `poisson_mg_object%initialize` and `solve`) is fed to the check that replaces it and must
!< return its code, with a message that starts with the code's name; a valid setup next to each must pass. The
!< isolated-mesh check is collective: a 2:1 AMR mesh whose refined block touches the boundary is refused on every
!< rank, a mesh refined only inside is accepted (run at any rank count).
#ifndef ELISHA_NO_MPI
use mpi
#endif
use, intrinsic :: ieee_exceptions, only : ieee_usual, ieee_get_halting_mode, ieee_set_halting_mode
use elisha
implicit none
integer(I4P), parameter   :: KP=ELISHA_BC_PERIODIC, KD=ELISHA_BC_DIRICHLET, KN=ELISHA_BC_NEUMANN !< Kinds.
integer(I4P), parameter   :: KG=ELISHA_BC_DIRICHLET_GHOST, KZ=ELISHA_BC_ZERO_GHOSTS, KF=ELISHA_BC_FREE_GHOSTS !< Kinds.
integer(I4P), parameter   :: SP=ELISHA_PATH_SPECTRAL, HI=ELISHA_PATH_HIERARCHICAL !< Paths.
logical,      parameter   :: T=.true., FF=.false. !< Shorthands.
real(R_P),    parameter   :: CUBE(3)=[0.1_R_P, 0.1_R_P, 0.1_R_P] !< Cubic cells.
type(elisha_comm_object)  :: comm                !< Communication context.
integer(I4P)              :: failures, st, me, ntot !< Failed checks, status, rank, total.
character(:), allocatable :: msg                 !< Message.
logical                   :: halting(3)          !< FP halting modes.
integer                   :: ierr, world         !< MPI.

failures = 0
#ifndef ELISHA_NO_MPI
call ieee_get_halting_mode(ieee_usual, halting)
call ieee_set_halting_mode(ieee_usual, .false.) ! hwloc raises IEEE flags inside MPI_Init (see elisha_test_comm_mpi)
call MPI_INIT(ierr)
call ieee_set_halting_mode(ieee_usual, halting)
world = MPI_COMM_WORLD
#else
world = 0 ; halting = .false.
#endif
call comm%init(world, st, msg)
! elisha_check_bc (fft initialize_blocks, mg initialize)
call elisha_check_bc(bcs(KP, KP, KD, KN, KG, KG), [FF, FF, FF], [0, 0, 0], st, msg) ; call expect('bc valid', ELISHA_OK)
call elisha_check_bc(bcs(KP, KP, 9, KN, KP, KP), [FF, FF, FF], [0, 0, 0], st, msg)
call expect('unknown boundary kind', ELISHA_ERR_BC_UNKNOWN)
call elisha_check_bc(bcs(KP, KD, KP, KP, KP, KP), [FF, FF, FF], [0, 0, 0], st, msg)
call expect('one periodic face and one non-periodic face', ELISHA_ERR_BC_PERIODIC_PAIR)
call elisha_check_bc(bcs(KP, KP, KG, KD, KP, KP), [FF, FF, FF], [0, 0, 0], st, msg)
call expect('ghost-Dirichlet face mixed with a face kind', ELISHA_ERR_BC_GHOST_PAIR)
call elisha_check_bc(bcs(KP, KP, KP, KP, KZ, KD), [FF, FF, FF], [2, 2, 2], st, msg)
call expect('zero-ghost face mixed with another kind', ELISHA_ERR_BC_ZERO_GHOST_PAIR)
call elisha_check_bc(bcs(KF, KZ, KP, KP, KP, KP), [FF, FF, FF], [0, 4, 4], st, msg)
call expect('free-space ghosts need the wide D(G)', ELISHA_ERR_BC_FREE_GHOSTS)
call elisha_check_bc(bcs(KF, KZ, KP, KP, KP, KP), [FF, FF, FF], [4, 4, 4], st, msg) ; call expect('free ghosts, wide', ELISHA_OK)
call elisha_check_bc(bcs(KP, KP, KP, KP, 7, KD), [FF, FF, T], [0, 0, 0], st, msg) ; call expect('null axis skipped', ELISHA_OK)
! elisha_check_dg_order (fft initialize_blocks)
call elisha_check_dg_order(6, st, msg) ; call expect('dg_order 6', ELISHA_OK)
call elisha_check_dg_order(3, st, msg) ; call expect('dg_order is not 0, 2, 4, 6 or 8', ELISHA_ERR_DG_ORDER)
! elisha_check_isolated (facade setup, fft initialize/initialize_blocks, mg initialize)
call elisha_check_isolated(T, FF, 0, [FF, FF, FF], [FF, FF, FF], CUBE, st, msg) ; call expect('isolated valid', ELISHA_OK)
call elisha_check_isolated(T, T, 0, [FF, FF, FF], [FF, FF, FF], CUBE, st, msg)
call expect('isolated boundaries take no bc', ELISHA_ERR_ISOLATED_BC)
call elisha_check_isolated(T, FF, 4, [FF, FF, FF], [FF, FF, FF], CUBE, st, msg)
call expect('isolated boundaries take the compact operator only', ELISHA_ERR_ISOLATED_WIDE)
call elisha_check_isolated(T, FF, 0, [FF, FF, T], [FF, FF, FF], CUBE, st, msg)
call expect('isolated boundaries are 3-D only', ELISHA_ERR_ISOLATED_DIMENSION)
call elisha_check_isolated(T, FF, 0, [FF, FF, FF], [FF, T, FF], CUBE, st, msg)
call expect('isolated boundaries need non-periodic directions', ELISHA_ERR_ISOLATED_PERIODIC)
call elisha_check_isolated(T, FF, 0, [FF, FF, FF], [FF, FF, FF], [0.1_R_P, 0.1_R_P, 0.2_R_P], st, msg)
call expect('isolated boundaries need cubic cells', ELISHA_ERR_ISOLATED_CELLS)
call elisha_check_isolated(FF, T, 4, [FF, FF, T], [T, T, T], [1._R_P, 2._R_P, 3._R_P], st, msg)
call expect('not isolated: nothing checked', ELISHA_OK)
! elisha_check_periodicity (fft initialize, mg initialize)
call elisha_check_periodicity(bcs(KP, KP, KD, KD, KN, KN), [T, FF, FF], [FF, FF, FF], st, msg)
call expect('periodicity agrees', ELISHA_OK)
call elisha_check_periodicity(bcs(KP, KP, KD, KD, KN, KN), [T, T, FF], [FF, FF, FF], st, msg)
call expect('direction periodic in the grid and not in bc', ELISHA_ERR_BC_PERIODICITY)
call elisha_check_periodicity(bcs(KP, KP, KP, KP, KN, KN), [T, FF, FF], [FF, FF, FF], st, msg)
call expect('direction not periodic in the grid and periodic in bc', ELISHA_ERR_BC_PERIODICITY)
! elisha_check_unbounded (fft initialize, mg initialize)
call elisha_check_unbounded(SP, [T, T, FF], [FF, FF, T], st, msg) ; call expect('spectral, periodic + null', ELISHA_OK)
call elisha_check_unbounded(SP, [T, FF, T], [FF, FF, FF], st, msg)
call expect('fft: direction not periodic without bc', ELISHA_ERR_BC_MISSING)
call elisha_check_unbounded(HI, [FF, FF, FF], [FF, FF, FF], st, msg) ; call expect('hierarchical, all bounded', ELISHA_OK)
call elisha_check_unbounded(HI, [T, FF, T], [FF, FF, FF], st, msg)
call expect('mg: mixed periodic and non-periodic directions without bc', ELISHA_ERR_BC_MISSING)
! elisha_check_reach (fft initialize_blocks)
call elisha_check_reach([8, 8, 8], [FF, FF, FF], 8, T, st, msg) ; call expect('reach met', ELISHA_OK)
call elisha_check_reach([8, 6, 8], [FF, FF, FF], 8, T, st, msg)
call expect('fewer cells than the stencil''s reach', ELISHA_ERR_DG_REACH)
call elisha_check_reach([8, 6, 1], [FF, FF, T], 6, T, st, msg) ; call expect('reach, null axis skipped', ELISHA_OK)
! elisha_check_path (facade setup and solve, fft initialize_blocks and solve, mg solve)
call elisha_check_path(SP, T, 4, bcs(KF, KF, KP, KP, KP, KP), FF, FF, T, FF, st, msg) ; call expect('path valid', ELISHA_OK)
call elisha_check_path(SP, FF, 4, bcs(KP, KP, KP, KP, KP, KP), FF, FF, FF, FF, st, msg)
call expect('the wide D(G) needs a uniform grid', ELISHA_ERR_DG_AMR)
call elisha_check_path(HI, T, 2, bcs(KP, KP, KP, KP, KP, KP), FF, FF, FF, FF, st, msg)
call expect('the wide D(G) needs the FFT path', ELISHA_ERR_DG_AMR)
call elisha_check_path(HI, FF, 0, bcs(KF, KF, KP, KP, KP, KP), FF, FF, FF, FF, st, msg)
call expect('mg: free-space ghosts need the FFT solver', ELISHA_ERR_BC_FREE_GHOSTS)
call elisha_check_path(SP, T, 0, bcs(KP, KP, KP, KP, KP, KP), T, T, FF, FF, st, msg)
call expect('redundant is for the compact operator, not isolated', ELISHA_ERR_REDUNDANT)
call elisha_check_path(SP, T, 4, bcs(KZ, KZ, KP, KP, KP, KP), FF, T, T, FF, st, msg)
call expect('redundant, CG path', ELISHA_ERR_REDUNDANT)
call elisha_check_path(SP, T, 0, bcs(KD, KD, KP, KP, KP, KP), FF, FF, FF, T, st, msg)
call expect('boundary values need the multigrid path', ELISHA_ERR_GHOST_VALUES)
call elisha_check_path(HI, FF, 0, bcs(KP, KP, KP, KP, KP, KP), T, FF, FF, T, st, msg)
call expect('mg: ghost values given with isolated boundaries', ELISHA_ERR_GHOST_VALUES)
call elisha_check_path(HI, FF, 0, bcs(KD, KN, KG, KG, KP, KP), FF, FF, FF, T, st, msg) ; call expect('ghost values, mg', ELISHA_OK)
! elisha_check_ghost_count (mg solve)
call elisha_check_ghost_count(12, 12, st, msg) ; call expect('ghost count', ELISHA_OK)
call elisha_check_ghost_count(11, 12, st, msg) ; call expect('fewer ghost values than faces', ELISHA_ERR_GHOST_COUNT)
! elisha_check_isolated_mesh (mg initialize), collective
call isolated_mesh('refined block on the boundary', [0, 0, 0, 0], ELISHA_ERR_ISOLATED_BOUNDARY_LEAF)
call isolated_mesh('refined block inside', [1, 1, 1, 0], ELISHA_OK)
ntot = failures
me = comm%rank
#ifndef ELISHA_NO_MPI
call MPI_ALLREDUCE(MPI_IN_PLACE, ntot, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)
#endif
call comm%destroy(st)
#ifndef ELISHA_NO_MPI
call MPI_FINALIZE(ierr)
#endif
if (ntot > 0) error stop 1
if (me == 0) print '(A)', 'elisha_test_setup_checks_mpi: ok'
contains
   pure function bcs(xl, xh, yl, yh, zl, zh) result(b)
   !< bc(2,3) from the six face kinds.
   integer(I4P), intent(in) :: xl, xh, yl, yh, zl, zh !< Kinds.
   integer(I4P)             :: b(2,3)                 !< bc.

   b = reshape([xl, xh, yl, yh, zl, zh], [2, 3])
   endfunction bcs

   subroutine expect(what, code)
   !< The last check returned `code`; a failure's message starts with the code's name.
   character(*), intent(in) :: what !< ADAM's refusal (or the valid case).
   integer(I4P), intent(in) :: code !< Expected code.

   if (st /= code) then
      call fail(what//': '//elisha_status_name(st)//' instead of '//elisha_status_name(code))
   elseif (code /= ELISHA_OK) then
      if (.not.allocated(msg)) then
         call fail(what//': no message')
      elseif (index(msg, elisha_status_name(code)//': ') /= 1) then
         call fail(what//': message "'//msg//'"')
      endif
   endif
   endsubroutine expect

   subroutine isolated_mesh(what, refined, code)
   !< A 4x4x4 level-0 mesh (4 blocks per axis) with one block refined, dealt round-robin; the isolated-mesh check.
   character(*), intent(in)  :: what       !< Case.
   integer(I4P), intent(in)  :: refined(4) !< The refined block.
   integer(I4P), intent(in)  :: code       !< Expected code.
   type(elisha_mesh_object)  :: mesh       !< Mesh.
   integer(I4P), allocatable :: gl(:,:), mine(:,:) !< Global and local leaves.
   integer(I4P)              :: i, j, k, c !< Counters.

   allocate(gl(4,63+8)) ! filled in place: nvfortran 26.9 empties a list grown by reshape([gl, x]) from gl(4,0)
   c = 0
   do k=0, 3 ; do j=0, 3 ; do i=0, 3
      if (all([i, j, k, 0] == refined)) cycle
      c = c + 1 ; gl(:,c) = [i, j, k, 0]
   enddo ; enddo ; enddo
   do k=0, 1 ; do j=0, 1 ; do i=0, 1
      c = c + 1 ; gl(:,c) = [2 * refined(1) + i, 2 * refined(2) + j, 2 * refined(3) + k, 1]
   enddo ; enddo ; enddo
   allocate(mine(4,count([(mod(i - 1, comm%nranks) == comm%rank, i=1, c)])))
   j = 0
   do i=1, c
      if (mod(i - 1, comm%nranks) == comm%rank) then
         j = j + 1 ; mine(:,j) = gl(:,i)
      endif
   enddo
   call mesh%init([4, 4, 4], 1, [4, 4, 4], CUBE, [0._R_P, 0._R_P, 0._R_P], [FF, FF, FF], [FF, FF, FF], mine, st, msg)
   if (st == ELISHA_OK) call mesh%validate(comm, st, msg)
   if (st /= ELISHA_OK) then
      call fail(what//': the mesh itself is refused: '//msg)
      return
   endif
   call elisha_check_isolated_mesh(mesh, comm, st, msg)
   call expect(what, code)
   endsubroutine isolated_mesh

   subroutine fail(what)
   !< Report a failed check.
   character(*), intent(in) :: what !< What failed.

   failures = failures + 1
   print '(A,I0,A)', '  FAIL rank ', comm%rank, ': '//what
   endsubroutine fail
endprogram elisha_test_setup_checks_mpi
