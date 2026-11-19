!< ELISHA test: the mesh descriptor's checks on any number of ranks, one case per refusal.
program elisha_test_mesh_mpi
!< ELISHA test: the mesh descriptor's checks on any number of ranks, one case per refusal.
!<
!< Each case is a global leaf list dealt round-robin to the ranks; every rank must get the expected status. Valid:
!< uniform, 2:1 AMR, a 2-D quadtree (null z), a 2-level jump at a non-periodic domain face. Refused: levels (uniform
!< asked on AMR), a gap, a duplicate, a block inside another, an overlap that hides a gap (the volume matches), a 2-level jump across a face and across a periodic
!< wrap, the key limit, no block at all (validate); a zero count, a null axis with two cells, a coordinate out of
!< range, a negative level, coords not [4,nb] (init). On success the gathered list and its table hold every leaf.
#ifndef ELISHA_NO_MPI
use mpi
#endif
use, intrinsic :: ieee_exceptions, only : ieee_usual, ieee_get_halting_mode, ieee_set_halting_mode
use elisha, only : I4P, R_P, ELISHA_OK, ELISHA_ERR_MESH_SHAPE, ELISHA_ERR_MESH_COORDS, ELISHA_ERR_MESH_LEVELS, &
                   ELISHA_ERR_MESH_COVERAGE, ELISHA_ERR_MESH_BALANCE, ELISHA_ERR_MESH_KEY_LIMIT, ELISHA_ERR_SHAPE, &
                   elisha_comm_object, elisha_mesh_object, elisha_status_name
implicit none
type(elisha_comm_object)  :: comm                !< Communication context.
integer(I4P)              :: failures, stat, n, me !< Failed checks, status, total, rank.
integer(I4P), allocatable :: g(:,:), base(:,:)    !< Global leaf lists.
logical                   :: halting(3)           !< FP halting modes.
integer                   :: ierr, world          !< MPI.
character(:), allocatable :: msg                  !< Message.

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
call comm%init(world, stat, msg)
! valid: uniform level 1 on 2x2x2 level-0 blocks
g = uniform_list([2, 2, 2], 1)
call case('uniform, uniform asked', g, [4, 4, 4], [2, 2, 2], [.true., .true., .true.], [.false., .false., .false.], &
          ELISHA_OK, uniform=.true.)
! valid AMR: block (0,0,0) of 2x2x2 refined
base = amr_base()
call case('2:1 AMR', base, [4, 4, 4], [2, 2, 2], [.false., .false., .false.], [.false., .false., .false.], ELISHA_OK)
call case('uniform asked on AMR', base, [4, 4, 4], [2, 2, 2], [.false., .false., .false.], &
          [.false., .false., .false.], ELISHA_ERR_MESH_LEVELS, uniform=.true.)
! a gap, a duplicate, an overlap
call case('a gap', base(:,2:), [4, 4, 4], [2, 2, 2], [.false., .false., .false.], [.false., .false., .false.], &
          ELISHA_ERR_MESH_COVERAGE)
call case('a duplicate', reshape([base, base(:,3)], [4, size(base, 2) + 1]), [4, 4, 4], [2, 2, 2], &
          [.false., .false., .false.], [.false., .false., .false.], ELISHA_ERR_MESH_COVERAGE)
call case('a block inside another', reshape([base, [0, 0, 0, 0]], [4, size(base, 2) + 1]), [4, 4, 4], [2, 2, 2], &
          [.false., .false., .false.], [.false., .false., .false.], ELISHA_ERR_MESH_COVERAGE)
! an overlap that hides a gap: the parent (0,0,0) added, the level-0 block (1,1,1) removed (the volume matches)
g = reshape([pack(base, spread(.not.all(base == spread([1, 1, 1, 0], 2, size(base, 2)), dim=1), 1, 4)), &
             [0, 0, 0, 0]], [4, size(base, 2)])
call case('an overlap that hides a gap', g, [4, 4, 4], [2, 2, 2], [.false., .false., .false.], &
          [.false., .false., .false.], ELISHA_ERR_MESH_COVERAGE)
! a 2-level jump: the level-1 child (1,0,0) of block 0, next to level-0 block (1,0,0), refined again
call case('2-level jump', refine(base, [1, 0, 0, 1]), [4, 4, 4], [2, 2, 2], [.false., .false., .false.], &
          [.false., .false., .false.], ELISHA_ERR_MESH_BALANCE)
! a 2-D quadtree (null z), valid
call case('2-D quadtree', quad_base(), [8, 8, 1], [2, 2, 1], [.false., .false., .false.], [.false., .false., .true.], &
          ELISHA_OK)
! periodic wrap: x has 4 level-0 blocks; block 0 refined, its child 0 refined again: across x = 0 it faces block 3
g = refine(refine(reshape([0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0], [4, 4]), [0, 0, 0, 0]), [0, 0, 0, 1])
call case('2-level jump at a domain face', g, [4, 1, 1], [4, 1, 1], [.false., .false., .false.], &
          [.false., .true., .true.], ELISHA_OK)
call case('2-level jump across a periodic wrap', g, [4, 1, 1], [4, 1, 1], [.true., .false., .false.], &
          [.false., .true., .true.], ELISHA_ERR_MESH_BALANCE)
! the key limit: 2 blocks of 2^18 cells along x
call case('key limit', reshape([0, 0, 0, 0, 1, 0, 0, 0], [4, 2]), [2**18, 1, 1], [2, 1, 1], &
          [.false., .false., .false.], [.false., .true., .true.], ELISHA_ERR_MESH_KEY_LIMIT)
! no block anywhere
deallocate(g) ; allocate(g(4,0))
call case('no block', g, [4, 4, 4], [2, 2, 2], [.false., .false., .false.], [.false., .false., .false.], &
          ELISHA_ERR_MESH_COVERAGE)
! init refusals (every rank holds the bad block, so each rank's init sees it)
call init_case('zero cells per block', [0, 4, 4], [2, 2, 2], [.false., .false., .false.], &
               reshape([0, 0, 0, 0], [4, 1]), ELISHA_ERR_MESH_SHAPE)
call init_case('null axis with two cells', [4, 4, 2], [2, 2, 1], [.false., .false., .true.], &
               reshape([0, 0, 0, 0], [4, 1]), ELISHA_ERR_MESH_SHAPE)
call init_case('coordinate out of range', [4, 4, 4], [2, 2, 2], [.false., .false., .false.], &
               reshape([4, 0, 0, 1], [4, 1]), ELISHA_ERR_MESH_COORDS)
call init_case('negative level', [4, 4, 4], [2, 2, 2], [.false., .false., .false.], &
               reshape([0, 0, 0, -1], [4, 1]), ELISHA_ERR_MESH_COORDS)
call init_case('coords not [4,nb]', [4, 4, 4], [2, 2, 2], [.false., .false., .false.], &
               reshape([0, 0, 0], [3, 1]), ELISHA_ERR_SHAPE)
n = failures
#ifndef ELISHA_NO_MPI
call MPI_ALLREDUCE(MPI_IN_PLACE, n, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)
#endif
me = comm%rank
call comm%destroy(stat)
#ifndef ELISHA_NO_MPI
call MPI_FINALIZE(ierr)
#endif
if (n > 0) error stop 1
if (me == 0) print '(A)', 'elisha_test_mesh_mpi: ok'
contains
   subroutine case(what, gl, nijk, nblk0, periodic, null, expect, uniform)
   !< Deal gl round-robin, init and validate, compare the status (and, valid, the gathered list).
   character(*), intent(in)           :: what        !< Case.
   integer(I4P), intent(in)           :: gl(:,:)     !< Global leaves.
   integer(I4P), intent(in)           :: nijk(3)     !< Cells per block.
   integer(I4P), intent(in)           :: nblk0(3)    !< Blocks at level 0.
   logical,      intent(in)           :: periodic(3) !< Periodic axes.
   logical,      intent(in)           :: null(3)     !< Null axes.
   integer(I4P), intent(in)           :: expect      !< Expected status.
   logical,      intent(in), optional :: uniform     !< One level asked.
   type(elisha_mesh_object)           :: mesh        !< Mesh.
   integer(I4P), allocatable          :: mine(:,:)   !< This rank's blocks.
   integer(I4P)                       :: st, i, nloc !< Status, counter, local count.

   nloc = 0
   allocate(mine(4,size(gl, 2)))
   do i=1, size(gl, 2)
      if (mod(i - 1, comm%nranks) == comm%rank) then
         nloc = nloc + 1 ; mine(:,nloc) = gl(:,i)
      endif
   enddo
   call mesh%init(nijk, 2, nblk0, [0.5_R_P, 0.25_R_P, 1._R_P], [0._R_P, 0._R_P, 0._R_P], periodic, null, &
                  mine(:,1:nloc), st, msg)
   if (st /= ELISHA_OK) then
      call fail(what//': init '//elisha_status_name(st)//' '//msg)
      return
   endif
   call mesh%validate(comm, st, msg, uniform)
   if (st /= expect) then
      call fail(what//': '//elisha_status_name(st)//' instead of '//elisha_status_name(expect)//' '//msg)
   elseif (st == ELISHA_OK) then
      if (mesh%nleaf /= size(gl, 2)) call fail(what//': gathered leaf count')
      do i=1, size(gl, 2)
         if (mesh%leaf_of(gl(:,i)) == 0) call fail(what//': a leaf missing from the table')
      enddo
      if (mesh%leaf_of([0, 0, 0, 30]) /= 0) call fail(what//': a non-leaf in the table')
   endif
   call mesh%destroy
   endsubroutine case

   subroutine init_case(what, nijk, nblk0, null, mine, expect)
   !< init with a bad descriptor: the expected status, on every rank.
   character(*), intent(in) :: what      !< Case.
   integer(I4P), intent(in) :: nijk(3)   !< Cells per block.
   integer(I4P), intent(in) :: nblk0(3)  !< Blocks at level 0.
   logical,      intent(in) :: null(3)   !< Null axes.
   integer(I4P), intent(in) :: mine(:,:) !< This rank's blocks.
   integer(I4P), intent(in) :: expect    !< Expected status.
   type(elisha_mesh_object) :: mesh      !< Mesh.
   integer(I4P)             :: st        !< Status.

   call mesh%init(nijk, 2, nblk0, [1._R_P, 1._R_P, 1._R_P], [0._R_P, 0._R_P, 0._R_P], [.false., .false., .false.], &
                  null, mine, st, msg)
   if (st /= expect) call fail(what//': '//elisha_status_name(st)//' instead of '//elisha_status_name(expect))
   if (st /= ELISHA_OK .and. mesh%is_initialized) call fail(what//': initialized after a refusal')
   endsubroutine init_case

   pure function uniform_list(nb0, lv) result(gl)
   !< Every block of level lv.
   integer(I4P), intent(in)  :: nb0(3), lv !< Blocks at level 0, level.
   integer(I4P), allocatable :: gl(:,:)    !< Leaves.
   integer(I4P)              :: i, j, k, c !< Counters.

   allocate(gl(4,product(nb0) * 8**lv))
   c = 0
   do k=0, nb0(3)*2**lv-1 ; do j=0, nb0(2)*2**lv-1 ; do i=0, nb0(1)*2**lv-1
      c = c + 1 ; gl(:,c) = [i, j, k, lv]
   enddo ; enddo ; enddo
   endfunction uniform_list

   pure function amr_base() result(gl)
   !< 2x2x2 level-0 blocks, block (0,0,0) refined: 7 + 8 leaves.
   integer(I4P), allocatable :: gl(:,:) !< Leaves.

   gl = refine(uniform_list([2, 2, 2], 0), [0, 0, 0, 0])
   endfunction amr_base

   pure function quad_base() result(gl)
   !< A 2-D quadtree: 2x2 level-0 blocks, block (1,1) refined, its corner child (3,3) refined again (its level-2
   !< leaves face level-1 leaves and the domain faces only: 2:1).
   integer(I4P), allocatable :: gl(:,:) !< Leaves.

   gl = refine(refine(reshape([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], [4, 4]), [1, 1, 0, 0]), [3, 3, 0, 1])
   endfunction quad_base

   pure function refine(gl, c4) result(r)
   !< gl with leaf c4 replaced by its children (2 per active axis; a z extent of 1 block per level is null).
   integer(I4P), intent(in)  :: gl(:,:) !< Leaves.
   integer(I4P), intent(in)  :: c4(4)   !< Leaf to refine.
   integer(I4P), allocatable :: r(:,:)  !< Leaves.
   integer(I4P)              :: i, j, k, n, m, nz, ny !< Counters, children per axis.
   logical                   :: flat_z, flat_y  !< Null axes, guessed from the list (every k or j zero).

   flat_z = all(gl(3,:) == 0) ; flat_y = all(gl(2,:) == 0)
   nz = merge(0, 1, flat_z) ; ny = merge(0, 1, flat_y)
   allocate(r(4,size(gl, 2) - 1 + 2 * (ny + 1) * (nz + 1)))
   m = 0
   do n=1, size(gl, 2)
      if (all(gl(:,n) == c4)) cycle
      m = m + 1 ; r(:,m) = gl(:,n)
   enddo
   do k=0, nz ; do j=0, ny ; do i=0, 1
      m = m + 1
      r(:,m) = [2 * c4(1) + i, merge(0, 2 * c4(2) + j, flat_y), merge(0, 2 * c4(3) + k, flat_z), c4(4) + 1]
   enddo ; enddo ; enddo
   endfunction refine

   subroutine fail(what)
   !< Report a failed check.
   character(*), intent(in) :: what !< What failed.

   failures = failures + 1
   print '(A,I0,A)', '  FAIL rank ', comm%rank, ': '//what
   endsubroutine fail
endprogram elisha_test_mesh_mpi
