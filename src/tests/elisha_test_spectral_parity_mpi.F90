!< ELISHA test: the spectral solver is bitwise ADAM's (the verbatim oracle) on every path, on any number of ranks.
program elisha_test_spectral_parity_mpi
!< ELISHA test: the spectral solver is bitwise ADAM's (the verbatim oracle) on every path, on any number of ranks.
!<
!< Every case runs through both solvers in one program from the same packed source: ADAM's `adam_poisson_fft_object`
!< (ADAM 669d0bdd verbatim, over the shims `adam_grid_object`, `adam_field_object`, `adam_mpih_global`) and
!< `elisha_spectral_solver_object` (over `elisha_mesh_object` and `elisha_comm_object`). The potentials, every ghost
!< layer included, and the energies where both compute them, are compared by their bit patterns: one differing bit
!< fails. Both objects must also take the same path (`slab`, `pcg`, `redundant` flags). Cases:
!< - `setup`/`initialize` + `solve_packed`: periodic (cubic level 0, non-cubic level 1, 2-D null axis), every face-kind
!<   family (D/D, N/N, D/N, N/D, ghost-Dirichlet, a mix with a periodic axis), isolated, the wide D(G) p = 2..8
!<   (periodic, and with Dirichlet/Neumann faces), zero-ghost faces with D(G) p = 4 and 8 and free-space ghosts p = 4
!<   (the rank-0 PCG path);
!< - `initialize_blocks` + `solve_blocks`: the redundant root (gw 0 and 1), ghost-Dirichlet via `dirichlet` (gw 0 and
!<   ngc), and the rank-0 FFT path (forced with `axis_dg_order`, as the PCG's sub-problems are), periodic and isolated.
!< Rank 0 prints per case the path, the verdict and a decomposition-independent checksum of ADAM's potential (the
!< XOR over the cells of each value's bits rotated by its global position): equal checksums on np 1, 2, 3 show ADAM's
!< potential is the same on every rank count. ADAM's solver needs MPI: built with `-DELISHA_NO_MPI` the program
!< prints a skip line and succeeds. The FP traps of a debug build are off around `MPI_Init` only.
#ifndef ELISHA_NO_MPI
use mpi
use, intrinsic :: iso_fortran_env, only : int64
use, intrinsic :: ieee_exceptions, only : ieee_usual, ieee_get_halting_mode, ieee_set_halting_mode
use elisha, only : I4P, R_P, ELISHA_OK, ELISHA_BC_PERIODIC, ELISHA_BC_DIRICHLET, ELISHA_BC_NEUMANN,        &
                   ELISHA_BC_DIRICHLET_GHOST, ELISHA_BC_ZERO_GHOSTS, ELISHA_BC_FREE_GHOSTS, elisha_comm_object, &
                   elisha_mesh_object, elisha_spectral_solver_object
use adam_grid_object, only : grid_object
use adam_field_object, only : field_object
use adam_mpih_global, only : mpih
use adam_poisson_fft_object, only : poisson_fft_object
implicit none
real(R_P),    parameter  :: FACTOR=2.5_R_P !< Source factor.
integer(I4P), parameter  :: P=ELISHA_BC_PERIODIC, D=ELISHA_BC_DIRICHLET, NM=ELISHA_BC_NEUMANN      !< Kinds.
integer(I4P), parameter  :: G=ELISHA_BC_DIRICHLET_GHOST, Z=ELISHA_BC_ZERO_GHOSTS, F=ELISHA_BC_FREE_GHOSTS !< Kinds.
logical,      parameter  :: T_=.true., F_=.false. !< Shorthands.
real(R_P),    parameter  :: DX8(3)=0.125_R_P  !< The spacing of an 8-cell box of side 1.
type(elisha_comm_object) :: c                !< Communication context.
integer(I4P)             :: failures, nfail, stat, ncase !< Failed checks, over the ranks; status; cases.
integer                  :: ierr             !< MPI error status.
logical                  :: halting(3)       !< The FP halting modes of the build (debug: on).
character(:), allocatable :: msg             !< Message.
real(R_P)                :: t_start          !< Wall clock.

failures = 0 ; ncase = 0
call ieee_get_halting_mode(ieee_usual, halting)
call ieee_set_halting_mode(ieee_usual, .false.)
call MPI_INIT(ierr)
call ieee_set_halting_mode(ieee_usual, halting)
call mpih%initialize
call c%init(MPI_COMM_WORLD, stat, msg) ; if (stat /= ELISHA_OK) call fail('comm init: '//msg)
t_start = MPI_WTIME()
if (c%rank == 0) print '(A,T31,A,T43,A,T51,A,T63,A,T81,A)', 'case', 'path', 'bitwise', 'max|diff|', &
                        'checksum(ADAM phi)', 'energy bits (ADAM)'
! 1. periodic
call setup_case('periodic cubic L0', [4,4,4], [2,2,2], 0, [0.125_R_P, 0.125_R_P, 0.125_R_P], [T_,T_,T_], [F_,F_,F_], &
                2, .true.)
call setup_case('periodic non-cubic L1', [4,2,3], [1,2,1], 1, [0.1_R_P, 0.07_R_P, 0.13_R_P], [T_,T_,T_], [F_,F_,F_], &
                2, .true.)
call setup_case('periodic 2-D null z', [4,4,1], [2,3,1], 0, [0.05_R_P, 0.08_R_P, 1._R_P], [T_,T_,F_], [F_,F_,T_], &
                2, .true.)
! 2. face kinds (non-periodic box, 8^3 cells)
call setup_case('faces D/D', [4,4,4], [2,2,2], 0, DX8, [F_,F_,F_], [F_,F_,F_], 2, .true., bc=kinds(D,D, D,D, D,D))
call setup_case('faces N/N', [4,4,4], [2,2,2], 0, DX8, [F_,F_,F_], [F_,F_,F_], 2, .true., bc=kinds(NM,NM, NM,NM, NM,NM))
call setup_case('faces D/N (4n)', [4,4,4], [2,2,2], 0, DX8, [F_,F_,F_], [F_,F_,F_], 2, .true., &
                bc=kinds(D,NM, D,NM, D,NM))
call setup_case('faces N/D (4n)', [4,4,4], [2,2,2], 0, DX8, [F_,F_,F_], [F_,F_,F_], 2, .true., &
                bc=kinds(NM,D, NM,D, NM,D))
call setup_case('faces ghost-D', [4,4,4], [2,2,2], 0, DX8, [F_,F_,F_], [F_,F_,F_], 2, .true., bc=kinds(G,G, G,G, G,G))
call setup_case('faces mix P, D/D, N/D', [4,4,4], [2,2,2], 0, [0.1_R_P, 0.07_R_P, 0.13_R_P], [T_,F_,F_], &
                [F_,F_,F_], 2, .true., bc=kinds(P,P, D,D, NM,D))
! 3. isolated
call setup_case('isolated cubic', [4,4,4], [2,2,2], 0, DX8, [F_,F_,F_], [F_,F_,F_], 2, .true., isolated=.true.)
! 4. the wide D(G)
call setup_case('wide p=2 periodic', [4,4,4], [2,2,2], 0, DX8, [T_,T_,T_], [F_,F_,F_], 3, .true., dg_order=2)
call setup_case('wide p=4 periodic', [4,4,4], [2,2,2], 0, DX8, [T_,T_,T_], [F_,F_,F_], 3, .true., dg_order=4)
call setup_case('wide p=6 periodic', [4,4,4], [2,2,2], 0, DX8, [T_,T_,T_], [F_,F_,F_], 3, .true., dg_order=6)
call setup_case('wide p=8 periodic', [4,4,4], [2,2,2], 0, DX8, [T_,T_,T_], [F_,F_,F_], 3, .true., dg_order=8)
call setup_case('wide p=4 D/N, N/N, P', [4,4,4], [2,2,2], 0, [0.1_R_P, 0.07_R_P, 0.13_R_P], [F_,F_,T_], &
                [F_,F_,F_], 3, .true., bc=kinds(D,NM, NM,NM, P,P), dg_order=4)
call setup_case('wide p=8 D/D', [4,4,4], [2,2,2], 0, DX8, [F_,F_,F_], [F_,F_,F_], 3, .true., &
                bc=kinds(D,D, D,D, D,D), dg_order=8)
! 5. zero-ghost faces and free-space ghosts with D(G): the rank-0 PCG path
call setup_case('zero ghosts p=4', [4,4,4], [2,2,2], 0, DX8, [F_,F_,T_], [F_,F_,F_], 3, .true., &
                bc=kinds(Z,Z, Z,Z, P,P), dg_order=4)
call setup_case('zero ghosts p=8', [4,4,4], [2,2,2], 0, DX8, [F_,T_,T_], [F_,F_,F_], 3, .false., &
                bc=kinds(Z,Z, P,P, P,P), dg_order=8)
call setup_case('free ghosts p=4', [4,4,4], [2,2,2], 0, DX8, [F_,T_,T_], [F_,F_,F_], 3, .true., &
                bc=kinds(F,F, P,P, P,P), dg_order=4)
! 6. initialize_blocks + solve_blocks
call blocks_case('blocks redundant gw=0', [2,2,2], [4,4,4], 0, DX8, [F_,F_,F_], .false., 0, .false., redundant=.true.)
call blocks_case('blocks redundant ghostD gw=1', [2,2,2], [4,4,4], 0, DX8, [F_,F_,F_], .false., 1, .false., &
                 dirichlet=.true., redundant=.true.)
call blocks_case('blocks dirichlet gw=0', [2,2,2], [4,4,4], 0, DX8, [F_,F_,F_], .false., 0, .true., dirichlet=.true.)
call blocks_case('blocks dirichlet gw=ngc', [2,2,2], [4,4,4], 2, DX8, [F_,F_,F_], .false., 2, .true., dirichlet=.true.)
call blocks_case('blocks rank-0 fft periodic', [2,2,2], [4,4,4], 2, DX8, [F_,F_,F_], .false., 2, .true., &
                 axis_dg_order=[0,0,0])
call blocks_case('blocks rank-0 fft isolated', [2,2,2], [4,4,4], 2, DX8, [F_,F_,F_], .true., 2, .true., &
                 axis_dg_order=[0,0,0])
call blocks_case('blocks rank-0 fft D/N', [2,2,2], [4,4,4], 2, DX8, [F_,F_,F_], .false., 2, .true., &
                 bc=kinds(D,NM, NM,D, D,D), axis_dg_order=[0,0,0])
nfail = failures
call MPI_ALLREDUCE(MPI_IN_PLACE, nfail, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)
if (c%rank == 0) print '(A,I0,A,I0,A,F7.2,A)', 'elisha_test_spectral_parity_mpi: ', ncase, ' cases on ', c%nranks, &
   ' rank(s), ', MPI_WTIME() - t_start, ' s'
call c%destroy(stat)
call MPI_FINALIZE(ierr)
if (nfail > 0) error stop 1
if (mpih%myrank == 0) print '(A)', 'elisha_test_spectral_parity_mpi: ok (bitwise ADAM on every case)'

contains
   pure function kinds(lx, hx, ly, hy, lz, hz) result(bc)
   !< The kinds of the six faces as bc(2,3).
   integer(I4P), intent(in) :: lx, hx, ly, hy, lz, hz !< Low and high kinds per axis.
   integer(I4P)             :: bc(2,3)                !< Kinds.

   bc = reshape([lx, hx, ly, hy, lz, hz], [2, 3])
   endfunction kinds

   subroutine deal(nbl, mine, gid)
   !< The blocks of a lattice dealt round-robin: this rank's (i, j, k) and global ids.
   integer(I4P),              intent(in)    :: nbl(3)     !< Blocks per axis.
   integer(I4P), allocatable, intent(inout) :: mine(:,:)  !< This rank's blocks [3,nb].
   integer(I4P), allocatable, intent(inout) :: gid(:)     !< Their global ids [nb].
   integer(I4P)                             :: n, b, g    !< Counters.

   n = 0
   do g=0, product(nbl)-1
      if (mod(g, c%nranks) == c%rank) n = n + 1
   enddo
   if (allocated(mine)) deallocate(mine)
   if (allocated(gid)) deallocate(gid)
   allocate(mine(3,n), gid(n))
   b = 0
   do g=0, product(nbl)-1
      if (mod(g, c%nranks) /= c%rank) cycle
      b = b + 1
      mine(:,b) = [mod(g, nbl(1)), mod(g / nbl(1), nbl(2)), g / (nbl(1) * nbl(2))]
      gid(b) = g
   enddo
   endsubroutine deal

   subroutine pack_source(nijk, mine, buf)
   !< The packed interior source of this rank's blocks (block-major, i fastest): a smooth non-periodic function of the
   !< global cell, with a nonzero mean.
   integer(I4P),           intent(in)    :: nijk(3)   !< Cells per block.
   integer(I4P),           intent(in)    :: mine(:,:) !< This rank's blocks [3,nb].
   real(R_P), allocatable, intent(inout) :: buf(:)    !< Packed source.
   integer(I4P)                          :: b, i, j, k, n, gi, gj, gk !< Counters.

   if (allocated(buf)) deallocate(buf)
   allocate(buf(max(size(mine, dim=2) * product(nijk), 1)))
   buf = 0._R_P
   n = 0
   do b=1, size(mine, dim=2)
      do k=1, nijk(3)
         do j=1, nijk(2)
            do i=1, nijk(1)
               gi = mine(1,b) * nijk(1) + i - 1 ; gj = mine(2,b) * nijk(2) + j - 1 ; gk = mine(3,b) * nijk(3) + k - 1
               n = n + 1
               buf(n) = cos(0.7_R_P * gi + 1.3_R_P * gj + 0.4_R_P * gk) + &
                        0.3_R_P * sin(0.21_R_P * gi * gj - 0.5_R_P * gk) + 0.25_R_P
            enddo
         enddo
      enddo
   enddo
   endsubroutine pack_source

   subroutine setup_case(label, nijk, nblk0, level, dx0, periodic, null, ngc, with_energy, isolated, bc, dg_order)
   !< One case through ADAM's initialize + solve_packed and ELISHA's setup + solve_packed; compare.
   character(*),           intent(in) :: label       !< Case label.
   integer,                intent(in) :: nijk(3)     !< Cells per block.
   integer,                intent(in) :: nblk0(3)    !< Blocks per axis at level 0.
   integer,                intent(in) :: level       !< Level of every block.
   real(R_P),              intent(in) :: dx0(3)      !< Spacing at level 0.
   logical,                intent(in) :: periodic(3) !< Periodic axes.
   logical,                intent(in) :: null(3)     !< Null axes.
   integer,                intent(in) :: ngc         !< Ghost layers.
   logical,                intent(in) :: with_energy !< Compare the energy too.
   logical,      optional, intent(in) :: isolated    !< Isolated boundaries.
   integer(I4P), optional, intent(in) :: bc(2,3)     !< Kinds.
   integer,      optional, intent(in) :: dg_order    !< Operator order.
   type(grid_object)                   :: grid        !< ADAM's grid (shim).
   type(field_object)                  :: field       !< ADAM's field (shim).
   type(poisson_fft_object)            :: adam        !< ADAM's solver.
   type(elisha_mesh_object)            :: mesh        !< ELISHA's mesh.
   type(elisha_spectral_solver_object) :: eli         !< ELISHA's solver.
   integer(I4P), allocatable           :: mine(:,:), gid(:), c4(:,:) !< This rank's blocks.
   real(R_P),    allocatable           :: buf(:), pa(:,:,:,:), pe(:,:,:,:) !< Source, potentials.
   integer(I4P)                        :: nbl(3), nb, b, dgo !< Blocks, order.
   real(R_P)                           :: ea, ee     !< Energies.
   real(R_P),    parameter             :: EMIN(3)=[0.3_R_P, -0.2_R_P, 1._R_P] !< Domain corner.

   ncase = ncase + 1
   dgo = 0 ; if (present(dg_order)) dgo = dg_order
   nbl = merge(nblk0, nblk0 * 2**level, null)
   call deal(nbl, mine, gid)
   nb = size(gid)
   allocate(c4(4,nb))
   do b=1, nb
      c4(:,b) = [mine(:,b), int(level, I4P)]
   enddo
   call grid%set(nijk=int(nijk, I4P), ngc=int(ngc, I4P), nblk0=int(nblk0, I4P), dx0=dx0, emin=EMIN, &
                 periodic=periodic, null=null, maxlevel=int(level, I4P))
   call field%set(c4)
   call mesh%init(int(nijk, I4P), int(ngc, I4P), int(nblk0, I4P), dx0, EMIN, periodic, null, c4, stat, msg)
   if (stat /= ELISHA_OK) then
      call fail(label//': mesh init: '//msg) ; return
   endif
   call pack_source(int(nijk, I4P), mine, buf)
   allocate(pa(1-ngc:nijk(1)+ngc,1-ngc:nijk(2)+ngc,1-ngc:nijk(3)+ngc,max(nb,1)))
   allocate(pe(1-ngc:nijk(1)+ngc,1-ngc:nijk(2)+ngc,1-ngc:nijk(3)+ngc,max(nb,1)))
   pa = 0._R_P ; pe = 0._R_P ; ea = 0._R_P ; ee = 0._R_P
   call adam%initialize(grid=grid, field=field, isolated=isolated, bc=bc, dg_order=dgo)
   if (with_energy) then
      call adam%solve_packed(field=field, buf=buf, factor=FACTOR, phi=pa, energy=ea)
   else
      call adam%solve_packed(field=field, buf=buf, factor=FACTOR, phi=pa)
   endif
   call eli%setup(mesh, c, stat, msg, isolated=isolated, bc=bc, dg_order=int(dgo, I4P))
   if (stat /= ELISHA_OK) then
      call fail(label//': setup: '//msg) ; return
   endif
   if (with_energy) then
      call eli%solve_packed(mesh, buf, FACTOR, pe, stat, msg, energy=ee)
   else
      call eli%solve_packed(mesh, buf, FACTOR, pe, stat, msg)
   endif
   if (stat /= ELISHA_OK) then
      call fail(label//': solve: '//msg) ; return
   endif
   call compare(label, adam, eli, reshape(pa(:,:,:,1:nb), [size(pa(:,:,:,1:nb))]), &
                reshape(pe(:,:,:,1:nb), [size(pe(:,:,:,1:nb))]), gid, size(pa(:,:,:,1)), with_energy, ea, ee)
   call adam%destroy
   call eli%destroy
   call mesh%destroy
   endsubroutine setup_case

   subroutine blocks_case(label, nblk, nijk, ngc, dxyz, null, isolated, gw, with_energy, dirichlet, bc, dg_order, &
                          axis_dg_order, redundant)
   !< One case through both solvers' initialize_blocks + solve_blocks; compare.
   character(*),           intent(in) :: label            !< Case label.
   integer,                intent(in) :: nblk(3)          !< Blocks per axis.
   integer,                intent(in) :: nijk(3)          !< Cells per block.
   integer,                intent(in) :: ngc              !< Ghost layers of the solver (isolated: padding margin).
   real(R_P),              intent(in) :: dxyz(3)          !< Spacing.
   logical,                intent(in) :: null(3)          !< Null axes.
   logical,                intent(in) :: isolated         !< Isolated boundaries.
   integer,                intent(in) :: gw               !< Ghost layers of the output.
   logical,                intent(in) :: with_energy      !< Compare the energy too.
   logical,      optional, intent(in) :: dirichlet        !< Ghost-Dirichlet everywhere.
   integer(I4P), optional, intent(in) :: bc(2,3)          !< Kinds.
   integer(I4P), optional, intent(in) :: dg_order         !< Operator order.
   integer,      optional, intent(in) :: axis_dg_order(3) !< Operator per axis (forces the rank-0 path).
   logical,      optional, intent(in) :: redundant        !< Every rank solves.
   type(poisson_fft_object)            :: adam            !< ADAM's solver.
   type(elisha_spectral_solver_object) :: eli             !< ELISHA's solver.
   integer(I4P), allocatable           :: mine(:,:), gid(:), ado(:) !< This rank's blocks, axis orders.
   real(R_P),    allocatable           :: buf(:), oa(:), oe(:) !< Source, outputs.
   integer(I4P)                        :: nb, ncg       !< Blocks, cells per block with ghosts.
   real(R_P)                           :: ea, ee        !< Energies.

   ncase = ncase + 1
   call deal(int(nblk, I4P), mine, gid)
   nb = size(gid)
   call pack_source(int(nijk, I4P), mine, buf)
   ncg = product(nijk + 2 * gw)
   allocate(oa(max(nb*ncg,1)), oe(max(nb*ncg,1)))
   oa = 0._R_P ; oe = 0._R_P ; ea = 0._R_P ; ee = 0._R_P
   if (present(axis_dg_order)) then
      ado = int(axis_dg_order, I4P)
      call adam%initialize_blocks(nblk=int(nblk, I4P), nijk=int(nijk, I4P), ngc=int(ngc, I4P), dxyz=dxyz, level=0_I4P, &
                                  null_xyz=null, isolated=isolated, dirichlet=dirichlet, bc=bc, dg_order=dg_order, &
                                  axis_dg_order=ado, redundant=redundant)
      call eli%initialize_blocks(nblk=int(nblk, I4P), nijk=int(nijk, I4P), ngc=int(ngc, I4P), dxyz=dxyz, level=0_I4P, &
                                 null_xyz=null, isolated=isolated, comm=c, stat=stat, errmsg=msg, dirichlet=dirichlet, &
                                 bc=bc, dg_order=dg_order, axis_dg_order=ado, redundant=redundant)
   else
      call adam%initialize_blocks(nblk=int(nblk, I4P), nijk=int(nijk, I4P), ngc=int(ngc, I4P), dxyz=dxyz, level=0_I4P, &
                                  null_xyz=null, isolated=isolated, dirichlet=dirichlet, bc=bc, dg_order=dg_order, &
                                  redundant=redundant)
      call eli%initialize_blocks(nblk=int(nblk, I4P), nijk=int(nijk, I4P), ngc=int(ngc, I4P), dxyz=dxyz, level=0_I4P, &
                                 null_xyz=null, isolated=isolated, comm=c, stat=stat, errmsg=msg, dirichlet=dirichlet, &
                                 bc=bc, dg_order=dg_order, redundant=redundant)
   endif
   if (stat /= ELISHA_OK) then
      call fail(label//': initialize_blocks: '//msg) ; return
   endif
   if (with_energy) then
      call adam%solve_blocks(nbl=nb, coords=mine, buf=buf, factor=FACTOR, gw=int(gw, I4P), out=oa, energy=ea)
      call eli%solve_blocks(nb, mine, buf, FACTOR, int(gw, I4P), oe, stat, msg, energy=ee)
   else
      call adam%solve_blocks(nbl=nb, coords=mine, buf=buf, factor=FACTOR, gw=int(gw, I4P), out=oa)
      call eli%solve_blocks(nb, mine, buf, FACTOR, int(gw, I4P), oe, stat, msg)
   endif
   if (stat /= ELISHA_OK) then
      call fail(label//': solve_blocks: '//msg) ; return
   endif
   call compare(label, adam, eli, oa(1:nb*ncg), oe(1:nb*ncg), gid, ncg, with_energy, ea, ee)
   call adam%destroy
   call eli%destroy
   endsubroutine blocks_case

   subroutine compare(label, adam, eli, a, e, gid, ncg, with_energy, ea, ee)
   !< Bitwise comparison of ADAM's and ELISHA's packed potentials (and energies); the path both took; the checksum.
   character(*),                        intent(in) :: label       !< Case label.
   type(poisson_fft_object),            intent(in) :: adam        !< ADAM's solver.
   type(elisha_spectral_solver_object), intent(in) :: eli         !< ELISHA's solver.
   real(R_P),                           intent(in) :: a(:), e(:)  !< Potentials, block-major with ghosts.
   integer(I4P),                        intent(in) :: gid(:)      !< Global ids of the blocks.
   integer,                             intent(in) :: ncg         !< Cells per block, ghosts included.
   logical,                             intent(in) :: with_energy !< Energies given.
   real(R_P),                           intent(in) :: ea, ee      !< Energies.
   integer(int64), allocatable :: ba(:), be(:) !< Bit patterns.
   integer(int64)              :: h, pos      !< Checksum, global position.
   integer                     :: nbad(1), b, i, ierr_ !< Mismatches, counters.
   real(R_P)                   :: dmax(1)     !< Max |difference|.
   character(16)               :: path        !< Path taken.
   logical                     :: same_path   !< Both took the same path.

   ba = transfer(a, [0_int64], size(a)) ; be = transfer(e, [0_int64], size(e))
   nbad = count(ba /= be)
   dmax = 0._R_P ; if (size(a) > 0) dmax = maxval(abs(a - e))
   if (with_energy) then
      if (transfer(ea, 0_int64) /= transfer(ee, 0_int64)) nbad = nbad + 1
      dmax = max(dmax, abs(ea - ee))
   endif
   h = 0_int64
   do b=1, size(gid)
      do i=1, ncg
         pos = int(gid(b), int64) * ncg + i
         h = ieor(h, ishftc(ieor(ba((b-1)*ncg+i), pos), int(mod(pos, 61_int64), kind(1))))
      enddo
   enddo
   call MPI_ALLREDUCE(MPI_IN_PLACE, nbad, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr_)
   call MPI_ALLREDUCE(MPI_IN_PLACE, dmax, 1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, ierr_)
   call MPI_ALLREDUCE(MPI_IN_PLACE, h, 1, MPI_INTEGER8, MPI_BXOR, MPI_COMM_WORLD, ierr_)
   if (eli%redundant) then
      path = 'redundant'
   elseif (eli%slab) then
      path = 'slab'
   elseif (eli%pcg) then
      path = 'rank-0 pcg'
   else
      path = 'rank-0 fft'
   endif
   same_path = (adam%slab .eqv. eli%slab) .and. (adam%pcg .eqv. eli%pcg) .and. (adam%redundant .eqv. eli%redundant)
   if (c%rank == 0) then
      if (with_energy) then
         print '(A,T31,A,T43,A,T51,ES10.2,T63,Z16.16,T81,Z16.16)', label, path, merge('yes', 'NO ', nbad(1) == 0), &
            dmax, h, transfer(ea, 0_int64)
      else
         print '(A,T31,A,T43,A,T51,ES10.2,T63,Z16.16,T81,A)', label, path, merge('yes', 'NO ', nbad(1) == 0), dmax, h, '-'
      endif
   endif
   if (nbad(1) /= 0) call fail(label//': not bitwise ADAM')
   if (.not.same_path) call fail(label//': ADAM and ELISHA took different paths')
   endsubroutine compare

   subroutine fail(what)
   !< Report a failed check.
   character(*), intent(in) :: what !< What failed.

   failures = failures + 1
   print '(A,I0,A)', '  FAIL rank ', c%rank, ': '//what
   endsubroutine fail
#else
implicit none
print '(A)', 'elisha_test_spectral_parity_mpi: skipped (ADAM''s FFT solver needs MPI; built with -DELISHA_NO_MPI)'
#endif
endprogram elisha_test_spectral_parity_mpi
