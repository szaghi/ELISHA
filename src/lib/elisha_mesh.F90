!< ELISHA, the mesh descriptor: a domain of uniform blocks, refined as a 2:1-balanced octree (quadtree on a null axis).
module elisha_mesh
!< ELISHA, the mesh descriptor: a domain of uniform blocks, refined as a 2:1-balanced octree (quadtree on a null axis).
!<
!< The contract that replaces ADAM's `grid` and `field` objects (audit: the solvers read about ten scalars of them and
!< the block coordinates). Plain data, given by the caller:
!< - `nijk(3)` cells per block, `ngc` ghost layers, `nblk0(3)` blocks per axis at level 0, `dx0(3)` the cell spacing at
!<   level 0, `emin(3)` the domain's lower corner, `periodic(3)`, `null(3)` (an axis without extent: 1 cell, 1 block);
!< - `coords(4,nb)`, this rank's blocks (ADAM's `field%coordinates`): (i, j, k) 0-based at the block's level l, so
!<   0 <= i_d < nblk0(d) 2^l on an active axis (0 on a null one), and l.
!<
!< `init` checks what one rank can (`ELISHA_ERR_MESH_SHAPE`, `ELISHA_ERR_MESH_COORDS`); `validate(comm)` checks the
!< whole mesh, collectively, with the same status on every rank:
!< - the levels: `lmin`, `lmax` over the ranks; one level when `uniform` (the spectral solver: `ELISHA_ERR_MESH_LEVELS`);
!< - the key limit: fewer than 2^19 cells along every axis at `lmax` (`ELISHA_ERR_MESH_KEY_LIMIT`);
!< - exact cover: no duplicate, no block inside another, and the blocks' volume the domain's (`ELISHA_ERR_MESH_COVERAGE`;
!<   disjoint blocks inside the domain with its volume cover it);
!< - 2:1 balance across faces: the leaf beyond each face of a leaf at level l is at level l or l - 1, or finer
!<   (`ELISHA_ERR_MESH_BALANCE`; across a periodic axis the neighbour wraps).
!< It keeps the gathered leaf list (`leaf_c`, `leaf_owner`, `leaf_loc`, in rank order) and its table (`leaf_hash`,
!< block key -> leaf id), which the hierarchical solver uses, as ADAM's multigrid gathered them.
use penf, only : I4P, I8P
use elisha_kind_parameters, only : R_P
use elisha_comm, only : elisha_comm_object
use elisha_keys, only : ELISHA_KEY_COORD_LIMIT, ELISHA_KEY_LEVEL_MAX, elisha_bkey, elisha_hash_table
use elisha_status, only : ELISHA_OK, ELISHA_ERR_MESH_SHAPE, ELISHA_ERR_MESH_COORDS, ELISHA_ERR_MESH_LEVELS, &
                          ELISHA_ERR_MESH_COVERAGE, ELISHA_ERR_MESH_BALANCE, ELISHA_ERR_MESH_KEY_LIMIT,     &
                          ELISHA_ERR_SHAPE, ELISHA_ERR_NOT_INITIALIZED, elisha_set_error
implicit none
private
public :: elisha_mesh_object

type :: elisha_mesh_object
   !< The mesh descriptor (see the module).
   integer(I4P)              :: nijk(3)=1_I4P       !< Cells per block per axis.
   integer(I4P)              :: ngc=0_I4P           !< Ghost layers.
   integer(I4P)              :: nblk0(3)=1_I4P      !< Blocks per axis at level 0.
   real(R_P)                 :: dx0(3)=0._R_P       !< Cell spacing at level 0.
   real(R_P)                 :: emin(3)=0._R_P      !< Domain lower corner.
   logical                   :: periodic(3)=.false. !< Periodic axes.
   logical                   :: null(3)=.false.     !< Null axes.
   integer(I4P)              :: nb=0_I4P            !< This rank's blocks.
   integer(I4P), allocatable :: coords(:,:)         !< This rank's blocks (i, j, k, level) [4,nb].
   logical                   :: is_initialized=.false. !< `init` succeeded.
   ! set by validate
   logical                   :: is_validated=.false. !< `validate` succeeded.
   integer(I4P)              :: lmin=0_I4P, lmax=0_I4P !< Leaf levels over the ranks.
   integer(I4P)              :: nleaf=0_I4P         !< Leaves over the ranks.
   integer(I4P), allocatable :: leaf_c(:,:)         !< Every leaf (i, j, k, level), rank order [4,nleaf].
   integer(I4P), allocatable :: leaf_owner(:)       !< Owner rank of each leaf.
   integer(I4P), allocatable :: leaf_loc(:)         !< Local block index of each leaf on its owner.
   type(elisha_hash_table)   :: leaf_hash           !< Block key -> leaf id.
   contains
      procedure, pass(self) :: init         !< Set the descriptor, check it locally.
      procedure, pass(self) :: validate     !< Check the whole mesh, collectively; gather the leaves.
      procedure, pass(self) :: destroy      !< Free the descriptor.
      procedure, pass(self) :: blocks_along !< Blocks per axis at a level.
      procedure, pass(self) :: cell_dx      !< Cell spacing at a level.
      procedure, pass(self) :: leaf_of      !< The leaf id of a block (0: no such leaf).
endtype elisha_mesh_object

contains
   subroutine init(self, nijk, ngc, nblk0, dx0, emin, periodic, null, coords, stat, errmsg)
   !< Set the descriptor and check what one rank can: the shape and this rank's block coordinates.
   class(elisha_mesh_object), intent(inout)         :: self        !< Mesh.
   integer(I4P),              intent(in)            :: nijk(3)     !< Cells per block per axis.
   integer(I4P),              intent(in)            :: ngc         !< Ghost layers.
   integer(I4P),              intent(in)            :: nblk0(3)    !< Blocks per axis at level 0.
   real(R_P),                 intent(in)            :: dx0(3)      !< Cell spacing at level 0.
   real(R_P),                 intent(in)            :: emin(3)     !< Domain lower corner.
   logical,                   intent(in)            :: periodic(3) !< Periodic axes.
   logical,                   intent(in)            :: null(3)     !< Null axes.
   integer(I4P),              intent(in)            :: coords(:,:) !< This rank's blocks (i, j, k, level) [4,nb].
   integer(I4P),              intent(out)           :: stat        !< Status.
   character(:), allocatable, intent(out), optional :: errmsg      !< Message.
   integer(I4P)                                     :: b, d        !< Counters.
   character(96)                                    :: w           !< Message buffer.

   call self%destroy
   if (any(nijk < 1) .or. ngc < 0 .or. any(nblk0 < 1) .or. any(.not.(dx0 > 0._R_P))) then
      write(w, '(A,3I0,A,I0,A,3(I0,1X))') 'cells per block ', nijk, ', ghosts ', ngc, ', blocks at level 0 ', nblk0
      call elisha_set_error(stat, errmsg, ELISHA_ERR_MESH_SHAPE, 'elisha_mesh_object%init: '//trim(w)// &
                            '(every count >= 1, ghosts >= 0, spacings > 0)')
      return
   endif
   do d=1, 3
      if (null(d) .and. (nijk(d) /= 1 .or. nblk0(d) /= 1)) then
         write(w, '(A,I0,A)') 'axis ', d, ' is null but has more than one cell or block'
         call elisha_set_error(stat, errmsg, ELISHA_ERR_MESH_SHAPE, 'elisha_mesh_object%init: '//trim(w))
         return
      endif
   enddo
   if (size(coords, dim=1) /= 4) then
      call elisha_set_error(stat, errmsg, ELISHA_ERR_SHAPE, 'elisha_mesh_object%init: coords is not [4,nb]')
      return
   endif
   self%nijk = nijk ; self%ngc = ngc ; self%nblk0 = nblk0 ; self%dx0 = dx0 ; self%emin = emin
   self%periodic = periodic ; self%null = null
   self%nb = size(coords, dim=2, kind=I4P)
   allocate(self%coords(4,self%nb))
   if (self%nb > 0) self%coords = coords
   do b=1, self%nb
      if (coords(4,b) < 0 .or. coords(4,b) > ELISHA_KEY_LEVEL_MAX) then
         write(w, '(A,I0,A,I0,A,I0)') 'block ', b, ' at level ', coords(4,b), ' outside 0..', ELISHA_KEY_LEVEL_MAX
         call elisha_set_error(stat, errmsg, ELISHA_ERR_MESH_COORDS, 'elisha_mesh_object%init: '//trim(w))
         call self%destroy
         return
      endif
      do d=1, 3
         if (coords(d,b) < 0 .or. int(coords(d,b), I8P) >= self%blocks_along(coords(4,b), d)) then
            write(w, '(A,I0,A,I0,A,I0,A,I0)') 'block ', b, ' coordinate ', coords(d,b), ' along axis ', d, &
                                              ' at level ', coords(4,b)
            call elisha_set_error(stat, errmsg, ELISHA_ERR_MESH_COORDS, 'elisha_mesh_object%init: '//trim(w)// &
                                  ' outside its level''s range')
            call self%destroy
            return
         endif
      enddo
   enddo
   self%is_initialized = .true.
   stat = ELISHA_OK
   endsubroutine init

   subroutine validate(self, comm, stat, errmsg, uniform)
   !< Check the whole mesh, collectively (every rank gets the same status), and gather the leaf list (see the module).
   class(elisha_mesh_object), intent(inout)         :: self    !< Mesh.
   type(elisha_comm_object),  intent(in)            :: comm    !< Communication context.
   integer(I4P),              intent(out)           :: stat    !< Status.
   character(:), allocatable, intent(out), optional :: errmsg  !< Message.
   logical,                   intent(in),  optional :: uniform !< One level needed (default: any 2:1 octree).
   integer(I4P)                                     :: lev(2), code(1), mpi_stat !< Level range, status agreement.
   integer(I4P), allocatable                        :: flat(:), counts(:) !< Gathered coordinates, counts per rank.
   integer(I8P)                                     :: vol(1), domain     !< Volumes, in cells of lmax blocks.
   integer(I4P)                                     :: b, d, r, nd, l, id !< Counters.
   integer(I4P)                                     :: a4(4)  !< An ancestor block (a named variable, not a
                                                              !< function result: no argument temporary).
   character(160)                                   :: w      !< Message buffer.
   character(:), allocatable                        :: mine   !< This rank's message.

   self%is_validated = .false.
   if (.not.self%is_initialized) then
      call elisha_set_error(stat, errmsg, ELISHA_ERR_NOT_INITIALIZED, 'elisha_mesh_object%validate: init first')
      return
   endif
   code = ELISHA_OK ; mine = ''
   ! levels over the ranks
   lev = huge(1_I4P)
   do b=1, self%nb
      lev = min(lev, [self%coords(4,b), -self%coords(4,b)])
   enddo
   call comm%min_i4(lev, mpi_stat) ; if (agree_mpi()) return
   self%lmin = lev(1) ; self%lmax = -lev(2)
   if (self%lmin > self%lmax) then ! no block on any rank
      call set_local(ELISHA_ERR_MESH_COVERAGE, 'the mesh has no block')
   elseif (present(uniform)) then
      if (uniform .and. self%lmin /= self%lmax) then
         write(w, '(A,I0,A,I0,A)') 'the blocks sit at levels ', self%lmin, ' to ', self%lmax, &
                                   '; one level is needed (the spectral solver)'
         call set_local(ELISHA_ERR_MESH_LEVELS, trim(w))
      endif
   endif
   if (agree()) return
   ! the key limit: cells along every axis at lmax below 2^19
   do d=1, 3
      if (self%blocks_along(self%lmax, d) * self%nijk(d) >= int(ELISHA_KEY_COORD_LIMIT, I8P)) then
         write(w, '(A,I0,A,I0,A,I0)') 'axis ', d, ' has ', self%blocks_along(self%lmax, d) * self%nijk(d), &
                                      ' cells at level ', self%lmax
         call set_local(ELISHA_ERR_MESH_KEY_LIMIT, trim(w)//' (2^19 or more)')
         exit
      endif
   enddo
   if (agree()) return
   ! the gathered leaf list and its table
   call comm%allgatherv_i4(reshape(self%coords, [4 * self%nb]), flat, counts, mpi_stat) ; if (agree_mpi()) return
   self%nleaf = size(flat, kind=I4P) / 4
   allocate(self%leaf_c(4,self%nleaf), self%leaf_owner(self%nleaf), self%leaf_loc(self%nleaf))
   if (self%nleaf > 0) self%leaf_c = reshape(flat, [4, self%nleaf])
   id = 0
   do r=1, size(counts)
      do b=1, counts(r) / 4
         id = id + 1 ; self%leaf_owner(id) = r - 1 ; self%leaf_loc(id) = b
      enddo
   enddo
   call self%leaf_hash%init(self%nleaf)
   do id=1, self%nleaf
      if (self%leaf_hash%get(elisha_bkey(self%leaf_c(:,id))) /= 0) then
         write(w, '(A,4(1X,I0))') 'a duplicate block (i, j, k, level):', self%leaf_c(:,id)
         call set_local(ELISHA_ERR_MESH_COVERAGE, trim(w))
         exit
      endif
      call self%leaf_hash%put(elisha_bkey(self%leaf_c(:,id)), id)
   enddo
   if (agree()) return
   ! exact cover: no block inside another (this rank's), then the volume (every rank sums the gathered list alike)
   do b=1, self%nb
      do l=self%lmin, self%coords(4,b)-1
         a4 = ancestor(self%coords(:,b), l)
         if (self%leaf_of(a4) /= 0) then
            write(w, '(A,4(1X,I0),A,I0)') 'block (i, j, k, level)', self%coords(:,b), ' lies inside a block at level ', l
            call set_local(ELISHA_ERR_MESH_COVERAGE, trim(w))
            exit
         endif
      enddo
      if (code(1) /= ELISHA_OK) exit
   enddo
   if (agree()) return
   nd = count(.not.self%null)
   vol = 0_I8P
   do id=1, self%nleaf
      vol(1) = vol(1) + 2_I8P**(nd * (self%lmax - self%leaf_c(4,id)))
   enddo
   domain = product(self%blocks_along(self%lmax, [1, 2, 3]))
   if (vol(1) /= domain) then
      write(w, '(A,I0,A,I0,A,I0)') 'the blocks fill ', vol(1), ' of the domain''s ', domain, ' blocks at level ', &
                                   self%lmax
      call set_local(ELISHA_ERR_MESH_COVERAGE, trim(w)//' (a gap)')
   endif
   if (agree()) return
   ! 2:1 balance across faces, this rank's leaves
   do b=1, self%nb
      call check_balance(self%coords(:,b))
      if (code(1) /= ELISHA_OK) exit
   enddo
   if (agree()) return
   self%is_validated = .true.
   stat = ELISHA_OK
   contains
      subroutine set_local(c, msg)
      !< Record this rank's first failure.
      integer(I4P), intent(in) :: c   !< Code.
      character(*), intent(in) :: msg !< Message.

      if (code(1) /= ELISHA_OK) return
      code = c ; mine = msg
      endsubroutine set_local

      function agree() result(failed)
      !< Agree on the status over the ranks (the largest code); true on a failure, with stat and errmsg set.
      logical      :: failed !< A rank failed.
      integer(I4P) :: c(1)   !< Code.

      c = code
      call comm%max_i4(c, mpi_stat)
      failed = c(1) /= ELISHA_OK .or. mpi_stat /= ELISHA_OK
      if (.not.failed) return
      if (mpi_stat /= ELISHA_OK) c = mpi_stat
      if (c(1) == code(1)) then
         call elisha_set_error(stat, errmsg, c(1), 'elisha_mesh_object%validate: '//mine)
      else
         call elisha_set_error(stat, errmsg, c(1), 'elisha_mesh_object%validate: on another rank')
      endif
      endfunction agree

      function agree_mpi() result(failed)
      !< A failed collective: the status of every rank that saw it.
      logical :: failed !< The collective failed.

      failed = mpi_stat /= ELISHA_OK
      if (failed) call elisha_set_error(stat, errmsg, mpi_stat, 'elisha_mesh_object%validate: a collective failed')
      endfunction agree_mpi

      subroutine check_balance(c4)
      !< The leaf beyond each face of the leaf c4 is at its level, one coarser, or finer.
      integer(I4P), intent(in) :: c4(4)  !< Leaf (i, j, k, level).
      integer(I4P)             :: nb4(4) !< Neighbour block at the leaf's level.
      integer(I4P)             :: dd, s, ll !< Axis, side, level.
      integer(I8P)             :: nmax   !< Blocks along the axis.

      do dd=1, 3
         if (self%null(dd)) cycle
         do s=-1, 1, 2
            nb4 = c4 ; nb4(dd) = c4(dd) + s
            nmax = self%blocks_along(c4(4), dd)
            if (nb4(dd) < 0 .or. nb4(dd) >= nmax) then
               if (.not.self%periodic(dd)) cycle ! a domain face
               nb4(dd) = int(modulo(int(nb4(dd), I8P), nmax), I4P)
            endif
            do ll=c4(4), self%lmin, -1
               a4 = ancestor(nb4, ll)
               if (self%leaf_of(a4) /= 0) then
                  if (ll < c4(4) - 1) then
                     write(w, '(A,4(1X,I0),A,I0,A,I0)') 'leaf (i, j, k, level)', c4, ' faces a leaf at level ', ll, &
                                                        ' across axis ', dd
                     call set_local(ELISHA_ERR_MESH_BALANCE, trim(w)//' (not 2:1)')
                  endif
                  exit
               endif
            enddo
            if (code(1) /= ELISHA_OK) return
         enddo
      enddo
      endsubroutine check_balance
   endsubroutine validate

   subroutine destroy(self)
   !< Free the descriptor.
   class(elisha_mesh_object), intent(inout) :: self !< Mesh.

   if (allocated(self%coords))     deallocate(self%coords)
   if (allocated(self%leaf_c))     deallocate(self%leaf_c)
   if (allocated(self%leaf_owner)) deallocate(self%leaf_owner)
   if (allocated(self%leaf_loc))   deallocate(self%leaf_loc)
   call self%leaf_hash%free
   self%nb = 0 ; self%nleaf = 0 ; self%lmin = 0 ; self%lmax = 0
   self%is_initialized = .false. ; self%is_validated = .false.
   endsubroutine destroy

   elemental function blocks_along(self, lv, d) result(n)
   !< Blocks along axis d at level lv: nblk0(d) 2^lv, nblk0(d) on a null axis.
   class(elisha_mesh_object), intent(in) :: self !< Mesh.
   integer(I4P),              intent(in) :: lv   !< Level.
   integer(I4P),              intent(in) :: d    !< Axis.
   integer(I8P)                          :: n    !< Blocks.

   n = int(self%nblk0(d), I8P)
   if (.not.self%null(d)) n = n * 2_I8P**lv
   endfunction blocks_along

   pure function cell_dx(self, lv) result(dx)
   !< Cell spacing at level lv: dx0 / 2^lv, dx0 on a null axis.
   class(elisha_mesh_object), intent(in) :: self  !< Mesh.
   integer(I4P),              intent(in) :: lv    !< Level.
   real(R_P)                             :: dx(3) !< Spacing.

   dx = merge(self%dx0, self%dx0 / 2._R_P**lv, self%null)
   endfunction cell_dx

   pure function leaf_of(self, c4) result(id)
   !< The leaf id of block c4 (i, j, k, level), 0 when it is not a leaf (after validate).
   class(elisha_mesh_object), intent(in) :: self  !< Mesh.
   integer(I4P),              intent(in) :: c4(4) !< Block.
   integer(I4P)                          :: id    !< Leaf id.

   id = self%leaf_hash%get(elisha_bkey(c4))
   endfunction leaf_of

   pure function ancestor(c4, lv) result(a4)
   !< The block at level lv (<= c4's) that contains c4; a null axis keeps its 0.
   integer(I4P), intent(in) :: c4(4) !< Block.
   integer(I4P), intent(in) :: lv    !< Level.
   integer(I4P)             :: a4(4) !< Ancestor.

   a4(1:3) = c4(1:3) / 2_I4P**(c4(4) - lv)
   a4(4) = lv
   endfunction ancestor
endmodule elisha_mesh
