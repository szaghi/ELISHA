!< ELISHA test: the hierarchical solver is bitwise ADAM's (the verbatim oracle) on every path, on any number of ranks.
program elisha_test_hierarchical_parity_mpi
!< ELISHA test: the hierarchical solver is bitwise ADAM's (the verbatim oracle) on every path, on any number of ranks.
!<
!< Every case runs through both solvers in one program from the same source: ADAM's `adam_poisson_mg_object` (ADAM
!< 669d0bdd verbatim, over the shims `adam_grid_object`, `adam_field_object`, `adam_mpih_global`; its root the verbatim
!< `adam_poisson_fft_object`) and `elisha_hierarchical_solver_object` (over `elisha_mesh_object` and
!< `elisha_comm_object`; its root `elisha_spectral_solver_object`). Compared by their bit patterns (one differing bit
!< fails), after a first solve and again after a warm-started second one (the source scaled by 1.25 less 0.1, the
!< boundary values halved, so the second solve iterates from the first one's potential): the potential with every ghost layer, the
!< energy, `iterations`, `iterations0`, `residual`, `tolerance_used`, `factor_mean`; once per case the boundary-face list
!< (`nbnd`, `bnd_row`, `bnd_w`, `bnd_kind`, `bnd_dir`, `bnd_x`), `symmetry_defect` of a fixed pair of vectors, and
!< `apply_a`, `apply_b`, `apply_bt` on fixed vectors. Both must take the same path: the root level `lbot`, `singular`,
!< and the root solver's `redundant` and `slab` flags, the redundant flag also against the case's expectation. Cases
!< (3-D blocks of 8^3 cells, 2-D of 8^2; the leaves dealt round-robin; a smooth source plus a constant):
!< - periodic: uniform with two levels inside the blocks (root capped at 64 cells); AMR, a corner block refined (its
!<   children meet level-0 leaves across the wrap); AMR, the centre block of 3^3 refined, root capped at 216 cells;
!<   uniform with the default root (one V-cycle is the exact solve), cubic and anisotropic cells;
!< - bounded: ghost-Dirichlet AMR (default kinds), with and without levels inside the blocks; the kinds Dirichlet,
!<   mixed Dirichlet/Neumann per axis and ghost-Dirichlet on AMR, with nonzero boundary values on every face
!<   (`ghost=`: B g and the inhomogeneous ghost fill); a Neumann-only box (singular), with boundary values;
!< - isolated (the James step): uniform, and AMR with the refined block inside;
!< - 2-D (null z) AMR, ghost-Dirichlet and periodic (levels inside the blocks);
!< - a distributed root: AMR on 5x4x4 level-0 blocks, the root 40x32x32 cells, above the redundant limit (32^3), so the
!<   V-cycle's root solve takes the spectral solver's slab path.
!< Rank 0 prints per case the path, the iterations of the two solves, the first residual's bits and the verdict.
!< ADAM's solver needs MPI: built with `-DELISHA_NO_MPI` the program prints a skip line and succeeds. The FP traps of
!< a debug build are off around `MPI_Init` only.
#ifndef ELISHA_NO_MPI
use mpi
use, intrinsic :: iso_fortran_env, only : int64
use, intrinsic :: ieee_exceptions, only : ieee_usual, ieee_get_halting_mode, ieee_set_halting_mode
use elisha, only : I4P, R_P, ELISHA_OK, ELISHA_BC_DIRICHLET, ELISHA_BC_NEUMANN, ELISHA_BC_DIRICHLET_GHOST, &
                   elisha_comm_object, elisha_mesh_object, elisha_hierarchical_solver_object
use adam_grid_object, only : grid_object
use adam_field_object, only : field_object
use adam_mpih_global, only : mpih
use adam_poisson_mg_object, only : poisson_mg_object
implicit none
real(R_P),    parameter  :: FACTOR=2.5_R_P      !< Source factor.
real(R_P),    parameter  :: TOL=1.e-11_R_P      !< FCG tolerance.
integer(I4P), parameter  :: MAX_IT=100_I4P      !< FCG iterations allowed.
integer(I4P), parameter  :: NG=2_I4P            !< Ghost layers.
integer(I4P), parameter  :: D=ELISHA_BC_DIRICHLET, NM=ELISHA_BC_NEUMANN, G=ELISHA_BC_DIRICHLET_GHOST !< Kinds.
logical,      parameter  :: T_=.true., F_=.false. !< Shorthands.
integer(I4P), parameter  :: N3(3)=[8_I4P, 8_I4P, 8_I4P], N2(3)=[8_I4P, 8_I4P, 1_I4P] !< Cells per block.
real(R_P),    parameter  :: H16(3)=1._R_P / 16._R_P, H24(3)=1._R_P / 24._R_P     !< Spacings of unit boxes.
type(elisha_comm_object) :: c                   !< Communication context.
integer(I4P)             :: failures, nfail, stat, ncase !< Failed checks, over the ranks; status; cases.
integer                  :: ierr                !< MPI error status.
logical                  :: halting(3)          !< The FP halting modes of the build (debug: on).
character(:), allocatable :: msg                !< Message.
real(R_P)                :: t_start             !< Wall clock.

failures = 0 ; ncase = 0
call ieee_get_halting_mode(ieee_usual, halting)
call ieee_set_halting_mode(ieee_usual, .false.)
call MPI_INIT(ierr)
call ieee_set_halting_mode(ieee_usual, halting)
call mpih%initialize
call c%init(MPI_COMM_WORLD, stat, msg) ; if (stat /= ELISHA_OK) call fail('comm init: '//msg)
t_start = MPI_WTIME()
if (c%rank == 0) print '(A,T26,A,T48,A,T62,A,T80,A,T88,A)', 'case', 'path (lbot, root)', 'iterations', &
                        'residual bits', 'bitwise', 'time'
! periodic
call run_case('uniform, root 64', [2,2,2], N3, H16, F_, [0,0,0], [F_,F_,F_], T_, F_, 64, T_)
call run_case('AMR corner (wrap)', [2,2,2], N3, H16, T_, [0,0,0], [F_,F_,F_], T_, F_, 0, T_)
call run_case('AMR centre, root 216', [3,3,3], N3, H24, T_, [1,1,1], [F_,F_,F_], T_, F_, 216, T_)
call run_case('uniform, exact root', [2,2,2], N3, H16, F_, [0,0,0], [F_,F_,F_], T_, F_, 0, T_)
call run_case('uniform aniso, exact', [2,3,1], N3, [0.1_R_P, 0.07_R_P, 0.13_R_P], F_, [0,0,0], [F_,F_,F_], T_, F_, 0, &
              T_)
! bounded
call run_case('AMR ghost-D (default)', [2,2,2], N3, H16, T_, [0,0,0], [F_,F_,F_], F_, F_, 0, T_)
call run_case('AMR ghost-D, root 64', [2,2,2], N3, H16, T_, [1,0,1], [F_,F_,F_], F_, F_, 64, T_)
call run_case('AMR D/D + g', [2,2,2], N3, H16, T_, [1,0,0], [F_,F_,F_], F_, F_, 0, T_, bc=kinds(D,D, D,D, D,D), &
              with_ghost=T_)
call run_case('AMR D/N,N/D,D/D + g', [2,2,2], N3, [0.1_R_P, 0.07_R_P, 0.13_R_P], T_, [0,1,0], [F_,F_,F_], F_, F_, 0, &
              T_, bc=kinds(D,NM, NM,D, D,D), with_ghost=T_)
call run_case('AMR ghost-D + g', [2,2,2], N3, H16, T_, [0,0,1], [F_,F_,F_], F_, F_, 0, T_, bc=kinds(G,G, G,G, G,G), &
              with_ghost=T_)
call run_case('AMR Neumann (sing.)+g', [2,2,2], N3, H16, T_, [1,1,0], [F_,F_,F_], F_, F_, 0, T_, &
              bc=kinds(NM,NM, NM,NM, NM,NM), with_ghost=T_)
call run_case('AMR Neumann (singular)', [2,2,2], N3, H16, T_, [1,1,0], [F_,F_,F_], F_, F_, 0, T_, &
              bc=kinds(NM,NM, NM,NM, NM,NM))
! isolated
call run_case('isolated uniform', [2,2,2], N3, H16, F_, [0,0,0], [F_,F_,F_], F_, T_, 0, T_)
call run_case('isolated AMR', [3,3,3], N3, H24, T_, [1,1,1], [F_,F_,F_], F_, T_, 0, T_)
! 2-D (null z)
call run_case('2-D AMR ghost-D', [2,2,1], N2, H16, T_, [0,1,0], [F_,F_,T_], F_, F_, 0, T_)
call run_case('2-D AMR periodic, r16', [2,2,1], N2, H16, T_, [0,1,0], [F_,F_,T_], T_, F_, 16, T_)
! a distributed root (above REDUNDANT_ROOT_MAX = 32^3 cells): the slab path inside the V-cycle
call run_case('AMR, slab root 40x32^2', [5,4,4], N3, H16, T_, [2,1,2], [F_,F_,F_], T_, F_, 0, F_)
nfail = failures
call MPI_ALLREDUCE(MPI_IN_PLACE, nfail, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)
if (c%rank == 0) print '(A,I0,A,I0,A,F7.2,A)', 'elisha_test_hierarchical_parity_mpi: ', ncase, ' cases on ', &
   c%nranks, ' rank(s), ', MPI_WTIME() - t_start, ' s'
call c%destroy(stat)
call MPI_FINALIZE(ierr)
if (nfail > 0) error stop 1
if (mpih%myrank == 0) print '(A)', 'elisha_test_hierarchical_parity_mpi: ok (bitwise ADAM on every case)'

contains
   pure function kinds(lx, hx, ly, hy, lz, hz) result(bc)
   !< The kinds of the six faces as bc(2,3).
   integer(I4P), intent(in) :: lx, hx, ly, hy, lz, hz !< Low and high kinds per axis.
   integer(I4P)             :: bc(2,3)                !< Kinds.

   bc = reshape([lx, hx, ly, hy, lz, hz], [2, 3])
   endfunction kinds

   subroutine run_case(label, nblk0, nijk, dx0, amr, refine, null, periodic, isolated, root_cells, redundant, bc, &
                       with_ghost)
   !< One case: the same mesh and source through both solvers, two solves, every comparison (see the program).
   character(*), intent(in)           :: label      !< Case label.
   integer,      intent(in)           :: nblk0(3)   !< Blocks per axis at level 0.
   integer(I4P), intent(in)           :: nijk(3)    !< Cells per block per axis.
   real(R_P),    intent(in)           :: dx0(3)     !< Cell spacing at level 0.
   logical,      intent(in)           :: amr        !< Refine block `refine` to level 1.
   integer,      intent(in)           :: refine(3)  !< The refined level-0 block (0-based).
   logical,      intent(in)           :: null(3)    !< Null axes.
   logical,      intent(in)           :: periodic   !< Every active axis periodic (else none).
   logical,      intent(in)           :: isolated   !< Isolated boundaries.
   integer,      intent(in)           :: root_cells !< Most cells of the root (0: the default).
   logical,      intent(in)           :: redundant  !< The root is expected redundant (every rank solves it).
   integer(I4P), intent(in), optional :: bc(2,3)    !< Kinds per face.
   logical,      intent(in), optional :: with_ghost !< Solve with nonzero boundary values.
   real(R_P),    parameter            :: EMIN(3)=[0.3_R_P, -0.2_R_P, 0.1_R_P] !< Domain lower corner.
   type(grid_object)                       :: grid  !< ADAM's grid (shim).
   type(field_object)                      :: field !< ADAM's field (shim).
   type(poisson_mg_object)                 :: adam  !< ADAM's multigrid.
   type(elisha_mesh_object)                :: mesh  !< ELISHA's mesh.
   type(elisha_hierarchical_solver_object) :: eli   !< ELISHA's hierarchical solver.
   integer(I4P), allocatable :: leaves(:,:), coords(:,:)       !< Every leaf, this rank's [4,*].
   real(R_P),    allocatable :: bmin(:,:), bmax(:,:)           !< This rank's block corners [3,nb].
   real(R_P),    allocatable :: q(:,:,:,:,:), buf(:)           !< Source: ADAM's layout and packed.
   real(R_P),    allocatable :: pa(:,:,:,:), pe(:,:,:,:)       !< Potentials.
   real(R_P),    allocatable :: x(:), y(:), ya(:), ye(:), gv(:), sa(:), se(:) !< Fixed vectors, products.
   real(R_P)                 :: dx(3), xc(3), len3(3), ea, ee, syma, syme !< Measures.
   integer(I4P)              :: nleaf, nchild, lmax, nb, b, i, j, k, o, n, ncell, nbad, it(4) !< Counters.
   logical                   :: ghost_, same_path            !< Boundary values given; both took one path.
   real(R_P)                 :: r1                           !< First residual (ELISHA's).
   real(R_P)                 :: t0                           !< Wall clock.

   ncase = ncase + 1
   t0 = MPI_WTIME()
   ghost_ = .false. ; if (present(with_ghost)) ghost_ = with_ghost
   ! every leaf: level-0 blocks i fastest, the refined one replaced by its children (active axes); filled in place
   nchild = 2**count(.not.null)
   nleaf = product(nblk0) + merge(nchild - 1, 0, amr)
   allocate(leaves(4,nleaf))
   n = 0
   do k=0, nblk0(3)-1
      do j=0, nblk0(2)-1
         do i=0, nblk0(1)-1
            if (amr .and. all([i, j, k] == refine)) then
               do o=0, 7
                  if (null(3) .and. ibits(o, 2, 1) == 1) cycle
                  n = n + 1
                  leaves(:,n) = [2*i + ibits(o, 0, 1), 2*j + ibits(o, 1, 1), merge(k, 2*k + ibits(o, 2, 1), null(3)), 1]
               enddo
            else
               n = n + 1
               leaves(:,n) = [i, j, k, 0]
            endif
         enddo
      enddo
   enddo
   lmax = merge(1, 0, amr)
   call grid%set(nijk=nijk, ngc=NG, nblk0=int(nblk0, I4P), dx0=dx0, emin=EMIN, periodic=periodic .and. .not.null, &
                 null=null, maxlevel=lmax)
   nb = 0
   do n=0, nleaf-1
      if (mod(n, c%nranks) == c%rank) nb = nb + 1
   enddo
   allocate(coords(4,nb), bmin(3,nb), bmax(3,nb))
   b = 0
   do n=0, nleaf-1
      if (mod(n, c%nranks) /= c%rank) cycle
      b = b + 1
      coords(:,b) = leaves(:,n+1)
      bmin(:,b) = grid%domain_emin + coords(1:3,b) * grid%block_dxyz(:,coords(4,b))
      bmax(:,b) = bmin(:,b) + grid%block_dxyz(:,coords(4,b))
   enddo
   call field%set(coords, emin=bmin, emax=bmax)
   call mesh%init(nijk, NG, int(nblk0, I4P), dx0, EMIN, periodic .and. .not.null, null, coords, stat, msg)
   if (stat /= ELISHA_OK) then
      call fail(label//': mesh init: '//msg)
      return
   endif
   ! the source, in ADAM's conservative layout and packed (block-major, i fastest)
   ncell = product(nijk)
   allocate(q(1:1,1-NG:nijk(1)+NG,1-NG:nijk(2)+NG,1-NG:nijk(3)+NG,1:max(nb,1)), buf(max(nb*ncell,1)))
   allocate(pa(1-NG:nijk(1)+NG,1-NG:nijk(2)+NG,1-NG:nijk(3)+NG,1:max(nb,1)))
   allocate(pe, mold=pa)
   q = 0._R_P ; buf = 0._R_P ; pa = 0._R_P ; pe = 0._R_P
   len3 = grid%domain_emax - grid%domain_emin
   n = 0
   do b=1, nb
      dx = grid%cell_dxyz(:,coords(4,b))
      do k=1, nijk(3)
         do j=1, nijk(2)
            do i=1, nijk(1)
               xc = (bmin(:,b) - grid%domain_emin + ([i, j, k] - 0.5_R_P) * dx) / len3 * 2._R_P * acos(-1._R_P)
               n = n + 1
               q(1,i,j,k,b) = 0.7_R_P + cos(xc(1)) + 0.5_R_P * cos(2._R_P * xc(2) + xc(3) + 0.3_R_P) + &
                              0.25_R_P * sin(3._R_P * xc(1) + xc(2) + 2._R_P * xc(3))
               buf(n) = q(1,i,j,k,b)
            enddo
         enddo
      enddo
   enddo
   ! the setups
   if (present(bc)) then
      if (root_cells > 0) then
         call adam%initialize(grid=grid, field=field, tolerance=TOL, max_iterations=MAX_IT, bc=bc, root_cells=root_cells)
         call eli%setup(mesh, c, stat, msg, tolerance=TOL, max_iterations=MAX_IT, bc=bc, root_cells=root_cells)
      else
         call adam%initialize(grid=grid, field=field, tolerance=TOL, max_iterations=MAX_IT, bc=bc)
         call eli%setup(mesh, c, stat, msg, tolerance=TOL, max_iterations=MAX_IT, bc=bc)
      endif
   else
      if (root_cells > 0) then
         call adam%initialize(grid=grid, field=field, tolerance=TOL, max_iterations=MAX_IT, isolated=isolated, &
                              root_cells=root_cells)
         call eli%setup(mesh, c, stat, msg, tolerance=TOL, max_iterations=MAX_IT, isolated=isolated, &
                        root_cells=root_cells)
      else
         call adam%initialize(grid=grid, field=field, tolerance=TOL, max_iterations=MAX_IT, isolated=isolated)
         call eli%setup(mesh, c, stat, msg, tolerance=TOL, max_iterations=MAX_IT, isolated=isolated)
      endif
   endif
   if (stat /= ELISHA_OK) then
      call fail(label//': setup: '//msg)
      return
   endif
   nbad = 0
   ! the boundary-face list
   if (adam%nbnd /= eli%nbnd) then
      nbad = nbad + 1
   elseif (adam%nbnd > 0) then
      nbad = nbad + count(adam%bnd_row(1:adam%nbnd) /= eli%bnd_row(1:eli%nbnd))
      nbad = nbad + count(adam%bnd_kind(1:adam%nbnd) /= eli%bnd_kind(1:eli%nbnd))
      nbad = nbad + count(adam%bnd_dir(1:adam%nbnd) /= eli%bnd_dir(1:eli%nbnd))
      nbad = nbad + differ(adam%bnd_w(1:adam%nbnd), eli%bnd_w(1:eli%nbnd))
      nbad = nbad + differ(reshape(adam%bnd_x(:,1:adam%nbnd), [3*adam%nbnd]), &
                           reshape(eli%bnd_x(:,1:eli%nbnd), [3*eli%nbnd]))
   endif
   ! the boundary values: nonzero on every face, from where each lives
   allocate(gv(max(adam%nbnd,1)))
   gv = 0._R_P
   do i=1, adam%nbnd
      gv(i) = 0.3_R_P + sin(3._R_P * adam%bnd_x(1,i) + 0.4_R_P) * cos(2._R_P * adam%bnd_x(2,i)) + &
              0.2_R_P * adam%bnd_x(3,i)
   enddo
   ! two solves: the first from zero, the second warm-started from the first on a changed source (and boundary values)
   do o=1, 2
      if (o == 2) then
         q = 1.25_R_P * q - 0.1_R_P ; buf = 1.25_R_P * buf - 0.1_R_P ; gv = 0.5_R_P * gv
      endif
      if (ghost_) then
         call adam%solve(field=field, q=q, iv=1_I4P, factor=FACTOR, phi=pa, energy=ea, ghost=gv)
         call eli%solve_packed(mesh, buf, FACTOR, pe, stat, msg, energy=ee, ghost=gv)
      else
         call adam%solve(field=field, q=q, iv=1_I4P, factor=FACTOR, phi=pa, energy=ea)
         call eli%solve_packed(mesh, buf, FACTOR, pe, stat, msg, energy=ee)
      endif
      if (stat /= ELISHA_OK) then
         call fail(label//': solve: '//msg)
         return
      endif
      nbad = nbad + differ(reshape(pa, [size(pa)]), reshape(pe, [size(pe)]))
      nbad = nbad + differ([ea, adam%residual, adam%tolerance_used, adam%factor_mean], &
                           [ee, eli%residual, eli%tolerance_used, eli%factor_mean])
      if (adam%iterations /= eli%iterations .or. adam%iterations0 /= eli%iterations0) nbad = nbad + 1
      it(2*o-1) = eli%iterations0 ; it(2*o) = eli%iterations
      if (o == 1) r1 = eli%residual
   enddo
   ! the operators on fixed vectors
   n = max(nb*ncell,1)
   allocate(x(n), y(n), ya(n), ye(n), sa(max(adam%nbnd,1)), se(max(adam%nbnd,1)))
   do i=1, n
      x(i) = sin(1.3_R_P * i + 0.7_R_P * c%rank) ; y(i) = cos(2.1_R_P * i - 0.4_R_P * c%rank)
   enddo
   syma = adam%symmetry_defect(x, y)
   syme = eli%symmetry_defect(x, y, stat)
   nbad = nbad + differ([syma], [syme])
   ya = 0._R_P ; ye = 0._R_P
   call adam%apply_a(x, ya)
   call eli%apply_a(x, ye, stat)
   nbad = nbad + differ(ya, ye)
   ya = 0._R_P ; ye = 0._R_P
   call adam%apply_b(gv, ya)
   call eli%apply_b(gv, ye)
   nbad = nbad + differ(ya, ye)
   sa = 0._R_P ; se = 0._R_P
   call adam%apply_bt(x, sa)
   call eli%apply_bt(x, se)
   nbad = nbad + differ(sa, se)
   ! the path
   same_path = adam%lbot == eli%lbot .and. (adam%singular .eqv. eli%singular) .and. &
               (adam%fft%redundant .eqv. eli%fft%redundant) .and. (adam%fft%slab .eqv. eli%fft%slab)
   call MPI_ALLREDUCE(MPI_IN_PLACE, nbad, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)
   if (c%rank == 0) print '(A,T26,A,I0,A,A,T48,I0,A,I0,A,I0,A,I0,T62,Z16.16,T80,A,T86,F6.2,A)', label, 'lbot ', &
      eli%lbot, ', ', merge('redundant', 'slab     ', eli%fft%redundant), it(1), '+', it(2), ', ', it(3), '+', it(4), &
      transfer(r1, 0_int64), merge('yes', 'NO ', nbad == 0), MPI_WTIME() - t0, ' s'
   if (nbad /= 0) call fail(label//': not bitwise ADAM')
   if (.not.same_path) call fail(label//': ADAM and ELISHA took different paths')
   if (eli%fft%redundant .neqv. redundant) call fail(label//': the root path is not the expected one')
   call adam%destroy
   call eli%destroy
   call mesh%destroy
   endsubroutine run_case

   pure function differ(a, e) result(nbad)
   !< The entries of two real arrays whose bit patterns differ.
   real(R_P), intent(in) :: a(:), e(:) !< Arrays.
   integer(I4P)          :: nbad       !< Differing entries.

   nbad = count(transfer(a, [0_int64], size(a)) /= transfer(e, [0_int64], size(e)))
   endfunction differ

   subroutine fail(what)
   !< Report a failed check.
   character(*), intent(in) :: what !< What failed.

   failures = failures + 1
   print '(A,I0,A)', '  FAIL rank ', c%rank, ': '//what
   endsubroutine fail
#else
implicit none
print '(A)', 'elisha_test_hierarchical_parity_mpi: skipped (ADAM''s multigrid solver needs MPI; built with -DELISHA_NO_MPI)'
#endif
endprogram elisha_test_hierarchical_parity_mpi
