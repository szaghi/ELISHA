!< ELISHA test: the facade is bitwise ADAM's facade (the verbatim oracle) on every path, on any number of ranks.
program elisha_test_facade_parity_mpi
!< ELISHA test: the facade is bitwise ADAM's facade (the verbatim oracle) on every path, on any number of ranks.
!<
!< Every step runs through both facades in one program from the same mesh and source: ADAM's `flail_object`
!< (`adam_flail_object`, ADAM 669d0bdd verbatim, over the shims `adam_grid_object`, `adam_field_object`,
!< `adam_mpih_global` and the verbatim `adam_poisson_fft_object`, `adam_poisson_mg_object`) and ELISHA's
!< `elisha_solver_object` (over `elisha_mesh_object` and `elisha_comm_object`). The backend is the same integer for
!< both (`FLAIL_BACKEND_*` = `ELISHA_PATH_*`), and so are the kinds (`FLAIL_BC_*` = `ELISHA_BC_*`). A step is one setup
!< and two solves through `solve_packed`, the second warm-started on a changed source (scaled by 1.25 less 0.1, the
!< boundary values halved). Compared by their bit patterns (one differing bit fails): the chosen path (`is_multigrid`
!< against `is_hierarchical`, `path()`), the potential with every ghost layer, the energy (ADAM's facade multiplies the
!< FFT solver's by the cell volume, ELISHA's the spectral solver's), `iterations()` against the diagnostics; on the
!< multigrid path `mg%iterations0`, `mg%residual`, `mg%tolerance_used` and the boundary-face list (`mg%nbnd`,
!< `mg%bnd_kind`, `mg%bnd_dir`, `mg%bnd_x` against `boundary_faces`, the same list in the same order, whose values both
!< then take); on the FFT path `fft%pcg`, and with it `fft%iterations`, `fft%residual`. Cases (3-D blocks of 8^3 cells,
!< 3 ghost layers; the leaves dealt round-robin; a smooth source plus a constant):
!< - auto on a uniform mesh (FFT) and on an AMR one (multigrid); fft forced on a uniform mesh; multigrid forced on a
!<   uniform mesh (the root capped at 64 cells: levels inside the blocks) and on an AMR one;
!< - isolated: auto on a uniform mesh (FFT), multigrid forced on a uniform mesh, auto on AMR (the refined block inside);
!< - kinds: Dirichlet/Neumann mixed per axis and ghost-Dirichlet on the FFT path; Dirichlet/Neumann mixed and
!<   ghost-Dirichlet on the multigrid path (AMR, and forced on a uniform mesh), with boundary values;
!< - the wide D(G) of order 4 on the FFT path: periodic (direct), zero-ghost faces on x (the rank-0 PCG), free-space
!<   ghosts on x (PCG);
!< - a regrid sequence on one pair of facades: uniform (FFT), AMR (multigrid), AMR refined elsewhere (multigrid
!<   rebuilt), uniform again (FFT), each set up anew and solved twice.
!< Rank 0 prints per step the path, the iterations of the two solves and the verdict. ADAM's facade needs MPI: built with
!< `-DELISHA_NO_MPI` the program prints a skip line and succeeds. The FP traps of a debug build are off around `MPI_Init`.
#ifndef ELISHA_NO_MPI
use mpi
use, intrinsic :: iso_fortran_env, only : int64
use, intrinsic :: ieee_exceptions, only : ieee_usual, ieee_get_halting_mode, ieee_set_halting_mode
use elisha, only : I4P, R_P, ELISHA_OK, ELISHA_BC_PERIODIC, ELISHA_BC_DIRICHLET, ELISHA_BC_NEUMANN,                &
                   ELISHA_BC_DIRICHLET_GHOST, ELISHA_BC_ZERO_GHOSTS, ELISHA_BC_FREE_GHOSTS, ELISHA_PATH_AUTO,     &
                   ELISHA_PATH_SPECTRAL, ELISHA_PATH_HIERARCHICAL, elisha_comm_object, elisha_mesh_object,       &
                   elisha_solver_object, elisha_diagnostics_object
use adam_grid_object, only : grid_object
use adam_field_object, only : field_object
use adam_mpih_global, only : mpih
use adam_flail_object, only : flail_object, FLAIL_BACKEND_AUTO, FLAIL_BACKEND_FFT, FLAIL_BACKEND_MULTIGRID, &
                              FLAIL_BC_PERIODIC, FLAIL_BC_DIRICHLET, FLAIL_BC_NEUMANN, FLAIL_BC_DIRICHLET_GHOST, &
                              FLAIL_BC_ZERO_GHOSTS, FLAIL_BC_FREE_GHOSTS
implicit none
real(R_P),    parameter  :: FACTOR=2.5_R_P      !< Source factor.
integer(I4P), parameter  :: NG=3_I4P            !< Ghost layers.
integer(I4P), parameter  :: NIJK(3)=[8_I4P, 8_I4P, 8_I4P] !< Cells per block.
integer(I4P), parameter  :: AU=ELISHA_PATH_AUTO, SP=ELISHA_PATH_SPECTRAL, HI=ELISHA_PATH_HIERARCHICAL !< Paths.
integer(I4P), parameter  :: P=ELISHA_BC_PERIODIC, D=ELISHA_BC_DIRICHLET, NM=ELISHA_BC_NEUMANN !< Kinds.
integer(I4P), parameter  :: G=ELISHA_BC_DIRICHLET_GHOST, Z=ELISHA_BC_ZERO_GHOSTS, F=ELISHA_BC_FREE_GHOSTS !< Kinds.
logical,      parameter  :: T_=.true., F_=.false. !< Shorthands.
real(R_P),    parameter  :: H16(3)=1._R_P / 16._R_P, H24(3)=1._R_P / 24._R_P !< Spacings of unit boxes.
real(R_P),    parameter  :: HAN(3)=[0.1_R_P, 0.07_R_P, 0.13_R_P]             !< Anisotropic spacing.
type(elisha_comm_object) :: c                   !< Communication context.
type(flail_object)       :: adam                !< ADAM's facade (the regrid sequence).
type(elisha_solver_object) :: eli               !< ELISHA's facade (the regrid sequence).
integer(I4P)             :: failures, nfail, stat, nstep !< Failed checks, over the ranks; status; steps.
integer                  :: ierr                !< MPI error status.
logical                  :: halting(3)          !< The FP halting modes of the build (debug: on).
character(:), allocatable :: msg                !< Message.
real(R_P)                :: t_start             !< Wall clock.

failures = 0 ; nstep = 0
call ieee_get_halting_mode(ieee_usual, halting)
call ieee_set_halting_mode(ieee_usual, .false.)
call MPI_INIT(ierr)
call ieee_set_halting_mode(ieee_usual, halting)
call mpih%initialize
call c%init(MPI_COMM_WORLD, stat, msg) ; if (stat /= ELISHA_OK) call fail('comm init: '//msg)
t_start = MPI_WTIME()
! the constants the two facades share
if (any([FLAIL_BACKEND_AUTO, FLAIL_BACKEND_FFT, FLAIL_BACKEND_MULTIGRID] /= [AU, SP, HI])) &
   call fail('FLAIL_BACKEND_* and ELISHA_PATH_* differ')
if (any([FLAIL_BC_PERIODIC, FLAIL_BC_DIRICHLET, FLAIL_BC_NEUMANN, FLAIL_BC_DIRICHLET_GHOST, FLAIL_BC_ZERO_GHOSTS,  &
         FLAIL_BC_FREE_GHOSTS] /= [P, D, NM, G, Z, F])) call fail('FLAIL_BC_* and ELISHA_BC_* differ')
if (c%rank == 0) print '(A,T30,A,T44,A,T62,A,T70,A)', 'step', 'path', 'iterations', 'bitwise', 'time'
! the paths
call one('auto, uniform', [2,2,2], H16, F_, [0,0,0], [T_,T_,T_], AU, want=SP)
call one('auto, AMR', [2,2,2], H16, T_, [0,0,0], [T_,T_,T_], AU, want=HI)
call one('fft, uniform', [2,2,2], H16, F_, [0,0,0], [T_,T_,T_], SP, want=SP)
call one('multigrid, uniform, r64', [2,2,2], H16, F_, [0,0,0], [T_,T_,T_], HI, want=HI, root_cells=64_I4P)
call one('multigrid, AMR', [2,2,2], H16, T_, [1,0,1], [T_,T_,T_], HI, want=HI)
! isolated
call one('isolated, auto, uniform', [2,2,2], H16, F_, [0,0,0], [F_,F_,F_], AU, want=SP, isolated=T_)
call one('isolated, multigrid, uni.', [2,2,2], H16, F_, [0,0,0], [F_,F_,F_], HI, want=HI, isolated=T_)
call one('isolated, auto, AMR', [3,3,3], H24, T_, [1,1,1], [F_,F_,F_], AU, want=HI, isolated=T_)
! kinds
call one('fft D/N,N/D,D/D', [2,2,2], HAN, F_, [0,0,0], [F_,F_,F_], AU, want=SP, bc=kinds(D,NM, NM,D, D,D))
call one('fft ghost-D, y periodic', [2,2,2], H16, F_, [0,0,0], [F_,T_,F_], SP, want=SP, bc=kinds(G,G, P,P, G,G))
call one('mg D/N,N/D,D/D + g', [2,2,2], HAN, T_, [0,1,0], [F_,F_,F_], AU, want=HI, bc=kinds(D,NM, NM,D, D,D), &
         ghost=T_)
call one('mg ghost-D (default) + g', [2,2,2], H16, T_, [0,0,1], [F_,F_,F_], AU, want=HI, ghost=T_)
call one('mg forced, uni. D/D + g', [2,2,2], H16, F_, [0,0,0], [F_,T_,F_], HI, want=HI, bc=kinds(D,D, P,P, NM,D), &
         ghost=T_)
! the wide D(G)
call one('D(G) p=4 periodic', [2,2,2], H16, F_, [0,0,0], [T_,T_,T_], AU, want=SP, dg_order=4_I4P)
call one('D(G) p=4 zero ghosts (PCG)', [2,2,2], H16, F_, [0,0,0], [F_,T_,T_], AU, want=SP, bc=kinds(Z,Z, P,P, P,P), &
         dg_order=4_I4P)
call one('D(G) p=4 free ghosts (PCG)', [2,2,2], H16, F_, [0,0,0], [F_,T_,T_], SP, want=SP, bc=kinds(F,F, P,P, P,P), &
         dg_order=4_I4P)
! a regrid sequence on one pair of facades
call step('regrid 1: uniform', adam, eli, [2,2,2], H16, F_, [0,0,0], [T_,T_,T_], AU, want=SP)
call step('regrid 2: AMR corner', adam, eli, [2,2,2], H16, T_, [0,0,0], [T_,T_,T_], AU, want=HI)
call step('regrid 3: AMR elsewhere', adam, eli, [2,2,2], H16, T_, [1,1,0], [T_,T_,T_], AU, want=HI)
call step('regrid 4: uniform again', adam, eli, [2,2,2], H16, F_, [0,0,0], [T_,T_,T_], AU, want=SP)
call adam%destroy
call eli%destroy
nfail = failures
call MPI_ALLREDUCE(MPI_IN_PLACE, nfail, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)
if (c%rank == 0) print '(A,I0,A,I0,A,F7.2,A)', 'elisha_test_facade_parity_mpi: ', nstep, ' steps on ', c%nranks, &
   ' rank(s), ', MPI_WTIME() - t_start, ' s'
call c%destroy(stat)
call MPI_FINALIZE(ierr)
if (nfail > 0) error stop 1
if (mpih%myrank == 0) print '(A)', 'elisha_test_facade_parity_mpi: ok (bitwise ADAM''s facade on every step)'

contains
   pure function kinds(lx, hx, ly, hy, lz, hz) result(bc)
   !< The kinds of the six faces as bc(2,3).
   integer(I4P), intent(in) :: lx, hx, ly, hy, lz, hz !< Low and high kinds per axis.
   integer(I4P)             :: bc(2,3)                !< Kinds.

   bc = reshape([lx, hx, ly, hy, lz, hz], [2, 3])
   endfunction kinds

   subroutine one(label, nblk0, dx0, amr, refine, periodic, path, want, isolated, bc, dg_order, root_cells, ghost)
   !< A case of one step on a fresh pair of facades.
   character(*), intent(in)           :: label       !< Case label.
   integer,      intent(in)           :: nblk0(3)    !< Blocks per axis at level 0.
   real(R_P),    intent(in)           :: dx0(3)      !< Cell spacing at level 0.
   logical,      intent(in)           :: amr         !< Refine block `refine` to level 1.
   integer,      intent(in)           :: refine(3)   !< The refined level-0 block (0-based).
   logical,      intent(in)           :: periodic(3) !< Periodic axes.
   integer(I4P), intent(in)           :: path        !< Backend (path) asked.
   integer(I4P), intent(in)           :: want        !< Path expected.
   logical,      intent(in), optional :: isolated    !< Isolated boundaries.
   integer(I4P), intent(in), optional :: bc(2,3)     !< Kinds per face.
   integer(I4P), intent(in), optional :: dg_order    !< Operator order.
   integer(I4P), intent(in), optional :: root_cells  !< Most cells of the multigrid's root.
   logical,      intent(in), optional :: ghost       !< Solve with boundary values.
   type(flail_object)         :: a                   !< ADAM's facade.
   type(elisha_solver_object) :: e                   !< ELISHA's facade.

   call step(label, a, e, nblk0, dx0, amr, refine, periodic, path, want, isolated, bc, dg_order, root_cells, ghost)
   call a%destroy
   call e%destroy
   endsubroutine one

   subroutine step(label, a, e, nblk0, dx0, amr, refine, periodic, path, want, isolated, bc, dg_order, root_cells, ghost)
   !< One step: the same mesh and source through both facades, a setup and two solves, every comparison.
   character(*),               intent(in)           :: label       !< Step label.
   type(flail_object),         intent(inout)        :: a           !< ADAM's facade.
   type(elisha_solver_object), intent(inout)        :: e           !< ELISHA's facade.
   integer,                    intent(in)           :: nblk0(3)    !< Blocks per axis at level 0.
   real(R_P),                  intent(in)           :: dx0(3)      !< Cell spacing at level 0.
   logical,                    intent(in)           :: amr         !< Refine block `refine` to level 1.
   integer,                    intent(in)           :: refine(3)   !< The refined level-0 block (0-based).
   logical,                    intent(in)           :: periodic(3) !< Periodic axes.
   integer(I4P),               intent(in)           :: path        !< Backend (path) asked.
   integer(I4P),               intent(in)           :: want        !< Path expected.
   logical,                    intent(in), optional :: isolated    !< Isolated boundaries.
   integer(I4P),               intent(in), optional :: bc(2,3)     !< Kinds per face.
   integer(I4P),               intent(in), optional :: dg_order    !< Operator order.
   integer(I4P),               intent(in), optional :: root_cells  !< Most cells of the multigrid's root.
   logical,                    intent(in), optional :: ghost       !< Solve with boundary values.
   real(R_P),    parameter   :: EMIN(3)=[0.3_R_P, -0.2_R_P, 0.1_R_P] !< Domain lower corner.
   type(grid_object)               :: grid  !< ADAM's grid (shim).
   type(field_object)              :: field !< ADAM's field (shim).
   type(elisha_mesh_object)        :: mesh  !< ELISHA's mesh.
   type(elisha_diagnostics_object) :: dg    !< ELISHA's diagnostics.
   integer(I4P), allocatable :: leaves(:,:), coords(:,:), fk(:), fd(:) !< Leaves, this rank's; ELISHA's faces.
   real(R_P),    allocatable :: bmin(:,:), bmax(:,:), fx(:,:) !< This rank's block corners [3,nb]; face positions.
   real(R_P),    allocatable :: buf(:), pa(:,:,:,:), pe(:,:,:,:), gv(:) !< Source, potentials, boundary values.
   real(R_P)                 :: dx(3), xc(3), len3(3), ea, ee !< Measures, energies.
   integer(I4P)              :: nleaf, lmax, nb, b, i, j, k, o, n, nbad, it(4), nbnd !< Counters.
   logical                   :: ghost_, iso, hier !< Boundary values given; isolated; the multigrid path.
   real(R_P)                 :: t0                !< Wall clock.

   nstep = nstep + 1
   t0 = MPI_WTIME()
   ghost_ = .false. ; if (present(ghost)) ghost_ = ghost
   iso = .false. ; if (present(isolated)) iso = isolated
   ! every leaf: level-0 blocks i fastest, the refined one replaced by its 8 children
   nleaf = product(nblk0) + merge(7, 0, amr)
   allocate(leaves(4,nleaf))
   n = 0
   do k=0, nblk0(3)-1
      do j=0, nblk0(2)-1
         do i=0, nblk0(1)-1
            if (amr .and. all([i, j, k] == refine)) then
               do o=0, 7
                  n = n + 1
                  leaves(:,n) = [2*i + ibits(o, 0, 1), 2*j + ibits(o, 1, 1), 2*k + ibits(o, 2, 1), 1]
               enddo
            else
               n = n + 1
               leaves(:,n) = [i, j, k, 0]
            endif
         enddo
      enddo
   enddo
   lmax = merge(1, 0, amr)
   call grid%set(nijk=NIJK, ngc=NG, nblk0=int(nblk0, I4P), dx0=dx0, emin=EMIN, periodic=periodic, &
                 null=[F_, F_, F_], maxlevel=lmax)
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
   call mesh%init(NIJK, NG, int(nblk0, I4P), dx0, EMIN, periodic, [F_, F_, F_], coords, stat, msg)
   if (stat /= ELISHA_OK) then
      call fail(label//': mesh init: '//msg)
      return
   endif
   ! the source, packed (block-major, i fastest)
   allocate(buf(max(nb*product(NIJK),1)))
   allocate(pa(1-NG:NIJK(1)+NG,1-NG:NIJK(2)+NG,1-NG:NIJK(3)+NG,1:max(nb,1)))
   allocate(pe, mold=pa)
   buf = 0._R_P ; pa = 0._R_P ; pe = 0._R_P
   len3 = grid%domain_emax - grid%domain_emin
   n = 0
   do b=1, nb
      dx = grid%cell_dxyz(:,coords(4,b))
      do k=1, NIJK(3)
         do j=1, NIJK(2)
            do i=1, NIJK(1)
               xc = (bmin(:,b) - grid%domain_emin + ([i, j, k] - 0.5_R_P) * dx) / len3 * 2._R_P * acos(-1._R_P)
               n = n + 1
               buf(n) = 0.7_R_P + cos(xc(1)) + 0.5_R_P * cos(2._R_P * xc(2) + xc(3) + 0.3_R_P) + &
                        0.25_R_P * sin(3._R_P * xc(1) + xc(2) + 2._R_P * xc(3))
            enddo
         enddo
      enddo
   enddo
   ! the setups: the same backend, boundaries, operator and root cap
   call a%setup(grid=grid, field=field, isolated=iso, backend=path, bc=bc, dg_order=dg_order, root_cells=root_cells)
   call e%setup(mesh, c, stat, msg, path=path, isolated=isolated, bc=bc, dg_order=dg_order, root_cells=root_cells)
   if (stat /= ELISHA_OK) then
      call fail(label//': setup: '//msg)
      return
   endif
   nbad = 0
   hier = e%is_hierarchical()
   if (e%path() /= want) call fail(label//': the path is not the expected one')
   if (a%is_multigrid() .neqv. hier) then ! nothing more to compare (and ADAM's mg or fft is not the one in use)
      call fail(label//': is_multigrid and is_hierarchical differ')
      call mesh%destroy
      return
   endif
   ! the boundary-face list (multigrid path): ADAM's and boundary_faces, the same list in the same order
   nbnd = 0
   if (hier) then
      call e%boundary_faces(fk, fd, fx, stat, msg)
      if (stat /= ELISHA_OK) then
         call fail(label//': boundary_faces: '//msg)
         return
      endif
      nbnd = a%mg%nbnd
      if (size(fk) /= nbnd) then
         nbad = nbad + 1
      elseif (nbnd > 0) then
         nbad = nbad + count(a%mg%bnd_kind(1:nbnd) /= fk) + count(a%mg%bnd_dir(1:nbnd) /= fd)
         nbad = nbad + differ(reshape(a%mg%bnd_x(:,1:nbnd), [3*nbnd]), reshape(fx, [3*nbnd]))
      endif
   endif
   allocate(gv(max(nbnd,1)))
   gv = 0._R_P
   do i=1, nbnd
      gv(i) = 0.3_R_P + sin(3._R_P * a%mg%bnd_x(1,i) + 0.4_R_P) * cos(2._R_P * a%mg%bnd_x(2,i)) + &
              0.2_R_P * a%mg%bnd_x(3,i)
   enddo
   ! two solves: the first from zero, the second warm-started on a changed source (and boundary values)
   it = 0
   do o=1, 2
      if (o == 2) then
         buf = 1.25_R_P * buf - 0.1_R_P ; gv = 0.5_R_P * gv
      endif
      if (ghost_) then
         call a%solve_packed(field=field, buf=buf, factor=FACTOR, phi=pa, energy=ea, ghost=gv)
         call e%solve_packed(mesh, buf, FACTOR, pe, stat, msg, energy=ee, ghost=gv)
      else
         call a%solve_packed(field=field, buf=buf, factor=FACTOR, phi=pa, energy=ea)
         call e%solve_packed(mesh, buf, FACTOR, pe, stat, msg, energy=ee)
      endif
      if (stat /= ELISHA_OK) then
         call fail(label//': solve: '//msg)
         return
      endif
      dg = e%diagnostics()
      nbad = nbad + differ(reshape(pa, [size(pa)]), reshape(pe, [size(pe)]))
      nbad = nbad + differ([ea], [ee])
      if (dg%path /= want) nbad = nbad + 1
      if (hier) then
         if (a%iterations() /= dg%iterations .or. a%mg%iterations0 /= dg%iterations0) nbad = nbad + 1
         nbad = nbad + differ([a%mg%residual, a%mg%tolerance_used], [dg%residual, dg%tolerance_used])
         if (dg%pcg) nbad = nbad + 1
      else
         if (a%iterations() /= 0) nbad = nbad + 1
         if (a%fft%pcg .neqv. dg%pcg) nbad = nbad + 1
         if (dg%pcg) then
            if (a%fft%iterations /= dg%iterations) nbad = nbad + 1
            nbad = nbad + differ([a%fft%residual], [dg%residual])
         elseif (dg%iterations /= 0) then
            nbad = nbad + 1
         endif
      endif
      it(2*o-1) = dg%iterations0 ; it(2*o) = dg%iterations
   enddo
   call MPI_ALLREDUCE(MPI_IN_PLACE, nbad, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)
   if (c%rank == 0) print '(A,T30,A,T44,I0,A,I0,A,I0,A,I0,T62,A,T68,F6.2,A)', label, &
      merge('multigrid', 'fft      ', hier), it(1), '+', it(2), ', ', it(3), '+', it(4), &
      merge('yes', 'NO ', nbad == 0), MPI_WTIME() - t0, ' s'
   if (nbad /= 0) call fail(label//': not bitwise ADAM''s facade')
   call mesh%destroy
   endsubroutine step

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
print '(A)', 'elisha_test_facade_parity_mpi: skipped (ADAM''s facade needs MPI; built with -DELISHA_NO_MPI)'
#endif
endprogram elisha_test_facade_parity_mpi
