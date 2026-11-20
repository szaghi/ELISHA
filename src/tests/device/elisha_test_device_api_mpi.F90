!< ELISHA device test: the device-resident API (`on_device`) and the V-cycle root on the device are bitwise ADAM FNL's.
program elisha_test_device_api_mpi
!< ELISHA device test: the device-resident API (`on_device`) and the V-cycle root on the device are bitwise ADAM FNL's.
!<
!< M7, audit C4. Every case solves one source three ways, each on its own solver object (cold solves):
!< - A: ADAM's device twin (`poisson_mg_fnl_object` or `poisson_fft_fnl_object`, ADAM 669d0bdd verbatim, over the shims),
!<   host arrays; its V-cycle root goes through the host;
!< - E: ELISHA's device backend, host arrays (`on_device` absent); its root on the device (`root_solve`) when the root is
!<   the cuFFT solver's redundant path;
!< - D: the same backend with `buf` and `phi` present on the device (`!$acc enter data`), `on_device=.true.`, `phi`
!<   brought back with `!$acc update self` after the solve.
!< Compared by bit patterns (one differing bit fails): the potential with every ghost layer and the energy, E against A
!< and D against E; for the multigrid also the iterations (and `iterations0`) and the residual. Cases (8^3-cell blocks,
!< 2 ghost layers, the leaves dealt round-robin): hierarchical AMR periodic (a corner refined: the root periodic),
!< ghost-Dirichlet AMR (the root extended odd about the ghost centres), Dirichlet/Neumann AMR with boundary values (the
!< root extended over 2n and 4n cells), isolated AMR (the James step); spectral uniform periodic and D/N; and the facade
!< (`elisha_solver_object` with both device molds) on an AMR and a uniform mesh, D against E only. Rank 0 prints one
!< line per case and `elisha_test_device_api_mpi: ok` or `FAILED`. Built only in `tests-nvf-oac`; each rank takes the
!< GPU of its node-local rank; GPU-aware MPI above one rank.
use mpi
use, intrinsic :: iso_fortran_env, only : int64
use, intrinsic :: ieee_exceptions, only : ieee_usual, ieee_get_halting_mode, ieee_set_halting_mode
use openacc
use elisha, only : I4P, R_P, ELISHA_OK, ELISHA_BC_DIRICHLET, ELISHA_BC_NEUMANN, ELISHA_BC_DIRICHLET_GHOST, &
                   elisha_comm_object, elisha_mesh_object, elisha_solver_object, elisha_diagnostics_object
use adam_grid_object, only : grid_object
use adam_field_object, only : field_object
use adam_mpih_global, only : mpih
use adam_fnl_poisson_fft_object, only : poisson_fft_fnl_object
use adam_fnl_poisson_mg_object, only : poisson_mg_fnl_object
use elisha_device, only : elisha_spectral_solver_cufft_object, elisha_hierarchical_solver_oac_object
implicit none
real(R_P),    parameter  :: FACTOR=2.5_R_P      !< Source factor.
real(R_P),    parameter  :: TOL=1.e-11_R_P      !< FCG tolerance.
integer(I4P), parameter  :: MAX_IT=100_I4P      !< FCG iterations allowed.
integer(I4P), parameter  :: NG=2_I4P            !< Ghost layers.
integer(I4P), parameter  :: D=ELISHA_BC_DIRICHLET, NM=ELISHA_BC_NEUMANN, G=ELISHA_BC_DIRICHLET_GHOST !< Kinds.
integer(I4P), parameter  :: N3(3)=[8_I4P, 8_I4P, 8_I4P] !< Cells per block.
real(R_P),    parameter  :: H16(3)=1._R_P / 16._R_P, H24(3)=1._R_P / 24._R_P !< Spacings of unit boxes.
real(R_P),    parameter  :: EMIN(3)=[0.3_R_P, -0.2_R_P, 0.1_R_P] !< Domain lower corner.
logical,      parameter  :: T_=.true., F_=.false. !< Shorthands.
type(elisha_comm_object) :: c                   !< Communication context.
integer(I4P)             :: failures, nfail, stat, ncase !< Failed checks; status; cases.
integer                  :: ierr                !< MPI error status.
logical                  :: halting(3)          !< The FP halting modes of the build.
character(:), allocatable :: msg                !< Message.

failures = 0 ; ncase = 0
call ieee_get_halting_mode(ieee_usual, halting)
call ieee_set_halting_mode(ieee_usual, .false.)
call MPI_INIT(ierr)
call ieee_set_halting_mode(ieee_usual, halting)
call mpih%initialize
call c%init(MPI_COMM_WORLD, stat, msg) ; if (stat /= ELISHA_OK) call fail('comm init: '//msg)
call select_device
if (failures == 0) then
   call mg_case('MG AMR periodic', [2,2,2], H16, [0,0,0], T_, F_)
   call mg_case('MG AMR ghost-D (default)', [2,2,2], H16, [1,0,1], F_, F_)
   call mg_case('MG AMR D/N,N/D,D/D + g', [2,2,2], H16, [0,1,0], F_, F_, bc=kinds(D,NM, NM,D, D,D), with_ghost=T_)
   call mg_case('MG isolated AMR', [3,3,3], H24, [1,1,1], F_, T_)
   call fft_case('FFT periodic', [2,2,2], H16, T_)
   call fft_case('FFT D/N,N/D,D/D', [2,2,2], H16, F_, bc=kinds(D,NM, NM,D, D,D))
   call facade_case('facade AMR (hierarchical)', T_)
   call facade_case('facade uniform (spectral)', F_)
endif
nfail = failures
call MPI_ALLREDUCE(MPI_IN_PLACE, nfail, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)
if (c%rank == 0) then
   if (nfail == 0) then
      print '(A,I0,A,I0,A)', 'elisha_test_device_api_mpi: ok (', ncase, ' cases, ', c%nranks, ' rank(s): on_device '// &
            'and host arrays bitwise, both bitwise ADAM FNL)'
   else
      print '(A,I0,A)', 'elisha_test_device_api_mpi: FAILED (', nfail, ' checks)'
   endif
endif
call c%destroy(stat)
call MPI_FINALIZE(ierr)
if (nfail > 0) error stop 1

contains
   subroutine select_device
   !< Each rank on the GPU of its node-local rank modulo the node's GPUs; no GPU is a failure.
   integer :: local_comm, local_rank, nd, ierr_ !< Node communicator, node-local rank, GPUs, MPI error status.

   call MPI_COMM_SPLIT_TYPE(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, local_comm, ierr_)
   call MPI_COMM_RANK(local_comm, local_rank, ierr_)
   call MPI_COMM_FREE(local_comm, ierr_)
   nd = acc_get_num_devices(acc_device_nvidia)
   if (nd > 0) then
      call acc_set_device_num(mod(local_rank, nd), acc_device_nvidia)
      call acc_init(acc_device_nvidia)
   else
      call fail('no NVIDIA GPU visible to this rank')
   endif
   endsubroutine select_device

   pure function kinds(lx, hx, ly, hy, lz, hz) result(bc)
   !< The kinds of the six faces as bc(2,3).
   integer(I4P), intent(in) :: lx, hx, ly, hy, lz, hz !< Low and high kinds per axis.
   integer(I4P)             :: bc(2,3)                !< Kinds.

   bc = reshape([lx, hx, ly, hy, lz, hz], [2, 3])
   endfunction kinds

   subroutine make_mesh(nblk0, dx0, amr, refine, periodic, grid, field, mesh, buf, nb)
   !< The mesh (2:1 when `amr`: block `refine` replaced by its 8 children), ADAM's grid and field for it, and the packed
   !< source (block-major, i fastest): a constant plus smooth modes.
   integer,                  intent(in)    :: nblk0(3)  !< Blocks per axis at level 0.
   real(R_P),                intent(in)    :: dx0(3)    !< Cell spacing at level 0.
   logical,                  intent(in)    :: amr       !< Refine block `refine`.
   integer,                  intent(in)    :: refine(3) !< The refined block (0-based).
   logical,                  intent(in)    :: periodic  !< Every axis periodic (else none).
   type(grid_object),        intent(inout) :: grid      !< ADAM's grid.
   type(field_object),       intent(inout) :: field     !< ADAM's field.
   type(elisha_mesh_object), intent(inout) :: mesh      !< ELISHA's mesh.
   real(R_P), allocatable,   intent(out)   :: buf(:)    !< Packed source.
   integer(I4P),             intent(out)   :: nb        !< This rank's blocks.
   integer(I4P), allocatable :: leaves(:,:), coords(:,:) !< Every leaf, this rank's.
   real(R_P),    allocatable :: bmin(:,:), bmax(:,:)     !< Block corners.
   real(R_P)                 :: dx(3), xc(3), len3(3)    !< Measures.
   integer(I4P)              :: nleaf, b, i, j, k, o, n  !< Counters.

   nleaf = product(nblk0) + merge(7, 0, amr)
   allocate(leaves(4,nleaf))
   n = 0
   do k=0, nblk0(3)-1 ; do j=0, nblk0(2)-1 ; do i=0, nblk0(1)-1
      if (amr .and. all([i, j, k] == refine)) then
         do o=0, 7
            n = n + 1 ; leaves(:,n) = [2*i + ibits(o, 0, 1), 2*j + ibits(o, 1, 1), 2*k + ibits(o, 2, 1), 1]
         enddo
      else
         n = n + 1 ; leaves(:,n) = [i, j, k, 0]
      endif
   enddo ; enddo ; enddo
   call grid%set(nijk=N3, ngc=NG, nblk0=int(nblk0, I4P), dx0=dx0, emin=EMIN, periodic=[periodic, periodic, periodic], &
                 null=[F_, F_, F_], maxlevel=merge(1_I4P, 0_I4P, amr))
   nb = count([(mod(n, c%nranks) == c%rank, n=0, nleaf-1)])
   allocate(coords(4,nb), bmin(3,nb), bmax(3,nb))
   b = 0
   do n=0, nleaf-1
      if (mod(n, c%nranks) /= c%rank) cycle
      b = b + 1 ; coords(:,b) = leaves(:,n+1)
      bmin(:,b) = grid%domain_emin + coords(1:3,b) * grid%block_dxyz(:,coords(4,b))
      bmax(:,b) = bmin(:,b) + grid%block_dxyz(:,coords(4,b))
   enddo
   call field%set(coords, emin=bmin, emax=bmax)
   call mesh%init(N3, NG, int(nblk0, I4P), dx0, EMIN, [periodic, periodic, periodic], [F_, F_, F_], coords, stat, msg)
   if (stat /= ELISHA_OK) call fail('mesh init: '//msg)
   allocate(buf(max(nb*product(N3),1)))
   buf = 0._R_P ; len3 = grid%domain_emax - grid%domain_emin ; n = 0
   do b=1, nb
      dx = grid%cell_dxyz(:,coords(4,b))
      do k=1, N3(3) ; do j=1, N3(2) ; do i=1, N3(1)
         xc = (bmin(:,b) - grid%domain_emin + ([i, j, k] - 0.5_R_P) * dx) / len3 * 2._R_P * acos(-1._R_P)
         n = n + 1
         buf(n) = 0.7_R_P + cos(xc(1)) + 0.5_R_P * cos(2._R_P * xc(2) + xc(3) + 0.3_R_P) + &
                  0.25_R_P * sin(3._R_P * xc(1) + xc(2) + 2._R_P * xc(3))
      enddo ; enddo ; enddo
   enddo
   endsubroutine make_mesh

   subroutine mg_case(label, nblk0, dx0, refine, periodic, isolated, bc, with_ghost)
   !< A hierarchical case: A (ADAM FNL), E (host arrays), D (on_device); see the program.
   character(*), intent(in)           :: label      !< Case label.
   integer,      intent(in)           :: nblk0(3)   !< Blocks per axis at level 0.
   real(R_P),    intent(in)           :: dx0(3)     !< Cell spacing at level 0.
   integer,      intent(in)           :: refine(3)  !< The refined block.
   logical,      intent(in)           :: periodic   !< Every axis periodic.
   logical,      intent(in)           :: isolated   !< Isolated boundaries.
   integer(I4P), intent(in), optional :: bc(2,3)    !< Kinds.
   logical,      intent(in), optional :: with_ghost !< Boundary values.
   type(grid_object)                           :: grid  !< ADAM's grid.
   type(field_object)                          :: field !< ADAM's field.
   type(elisha_mesh_object)                    :: mesh  !< ELISHA's mesh.
   type(poisson_mg_fnl_object)                 :: a     !< ADAM FNL.
   type(elisha_hierarchical_solver_oac_object) :: e, dv !< ELISHA, host arrays and device arrays.
   real(R_P), allocatable :: buf(:), pa(:,:,:,:), pe(:,:,:,:), pd(:,:,:,:), gv(:) !< Source, potentials, values.
   real(R_P)              :: ea, ee, ed  !< Energies.
   integer(I4P)           :: nb, i, nbad !< Blocks, counter, differing bits.
   logical                :: ghost_      !< Boundary values given.

   ncase = ncase + 1
   ghost_ = .false. ; if (present(with_ghost)) ghost_ = with_ghost
   call make_mesh(nblk0, dx0, .true., refine, periodic, grid, field, mesh, buf, nb)
   allocate(pa(1-NG:N3(1)+NG,1-NG:N3(2)+NG,1-NG:N3(3)+NG,1:max(nb,1)))
   allocate(pe, mold=pa) ; allocate(pd, mold=pa)
   pa = 0._R_P ; pe = 0._R_P ; pd = 0._R_P
   if (present(bc)) then
      call a%initialize(grid=grid, field=field, tolerance=TOL, max_iterations=MAX_IT, bc=bc)
      call e%setup(mesh, c, stat, msg, tolerance=TOL, max_iterations=MAX_IT, bc=bc)
      if (stat == ELISHA_OK) call dv%setup(mesh, c, stat, msg, tolerance=TOL, max_iterations=MAX_IT, bc=bc)
   else
      call a%initialize(grid=grid, field=field, tolerance=TOL, max_iterations=MAX_IT, isolated=isolated)
      call e%setup(mesh, c, stat, msg, tolerance=TOL, max_iterations=MAX_IT, isolated=isolated)
      if (stat == ELISHA_OK) call dv%setup(mesh, c, stat, msg, tolerance=TOL, max_iterations=MAX_IT, isolated=isolated)
   endif
   if (stat /= ELISHA_OK) then
      call fail(label//': setup: '//msg) ; return
   endif
   allocate(gv(max(a%nbnd,1)))
   gv = 0._R_P
   do i=1, a%nbnd
      gv(i) = 0.3_R_P + sin(3._R_P * a%bnd_x(1,i) + 0.4_R_P) * cos(2._R_P * a%bnd_x(2,i)) + 0.2_R_P * a%bnd_x(3,i)
   enddo
   if (ghost_) then
      call a%solve_packed(field=field, buf=buf, factor=FACTOR, phi=pa, energy=ea, ghost=gv)
      call e%solve_packed(mesh, buf, FACTOR, pe, stat, msg, energy=ee, ghost=gv)
   else
      call a%solve_packed(field=field, buf=buf, factor=FACTOR, phi=pa, energy=ea)
      call e%solve_packed(mesh, buf, FACTOR, pe, stat, msg, energy=ee)
   endif
   if (stat /= ELISHA_OK) then
      call fail(label//': solve E: '//msg) ; return
   endif
   !$acc enter data copyin(buf, pd)
   if (ghost_) then
      call dv%solve_packed(mesh, buf, FACTOR, pd, stat, msg, energy=ed, ghost=gv, on_device=.true.)
   else
      call dv%solve_packed(mesh, buf, FACTOR, pd, stat, msg, energy=ed, on_device=.true.)
   endif
   !$acc update self(pd)
   !$acc exit data delete(buf, pd)
   if (stat /= ELISHA_OK) then
      call fail(label//': solve D: '//msg) ; return
   endif
   nbad = differ(flat(pe), flat(pa)) + differ([ee, e%residual], [ea, a%residual])
   if (e%iterations /= a%iterations .or. e%iterations0 /= a%iterations0) nbad = nbad + 1
   call report(label//': E vs ADAM FNL', nbad)
   nbad = differ(flat(pd), flat(pe)) + differ([ed, dv%residual], [ee, e%residual])
   if (dv%iterations /= e%iterations .or. dv%iterations0 /= e%iterations0) nbad = nbad + 1
   call report(label//': on_device vs host arrays', nbad)
   call a%destroy ; call e%destroy ; call dv%destroy ; call mesh%destroy
   endsubroutine mg_case

   subroutine fft_case(label, nblk0, dx0, periodic, bc)
   !< A spectral case: A (ADAM FNL), E (host arrays), D (on_device).
   character(*), intent(in)           :: label    !< Case label.
   integer,      intent(in)           :: nblk0(3) !< Blocks per axis.
   real(R_P),    intent(in)           :: dx0(3)   !< Cell spacing.
   logical,      intent(in)           :: periodic !< Every axis periodic.
   integer(I4P), intent(in), optional :: bc(2,3)  !< Kinds.
   type(grid_object)                         :: grid  !< ADAM's grid.
   type(field_object)                        :: field !< ADAM's field.
   type(elisha_mesh_object)                  :: mesh  !< ELISHA's mesh.
   type(poisson_fft_fnl_object)              :: a     !< ADAM FNL.
   type(elisha_spectral_solver_cufft_object) :: e, dv !< ELISHA, host arrays and device arrays.
   real(R_P), allocatable :: buf(:), pa(:,:,:,:), pe(:,:,:,:), pd(:,:,:,:) !< Source, potentials.
   real(R_P)              :: ea, ee, ed !< Energies.
   integer(I4P)           :: nb, nbad   !< Blocks, differing bits.

   ncase = ncase + 1
   call make_mesh(nblk0, dx0, .false., [0,0,0], periodic, grid, field, mesh, buf, nb)
   allocate(pa(1-NG:N3(1)+NG,1-NG:N3(2)+NG,1-NG:N3(3)+NG,1:max(nb,1)))
   allocate(pe, mold=pa) ; allocate(pd, mold=pa)
   pa = 0._R_P ; pe = 0._R_P ; pd = 0._R_P
   call a%initialize(grid=grid, field=field, bc=bc)
   call e%setup(mesh, c, stat, msg, bc=bc)
   if (stat == ELISHA_OK) call dv%setup(mesh, c, stat, msg, bc=bc)
   if (stat /= ELISHA_OK) then
      call fail(label//': setup: '//msg) ; return
   endif
   call a%solve_packed(field=field, buf=buf, factor=FACTOR, phi=pa, energy=ea)
   call e%solve_packed(mesh, buf, FACTOR, pe, stat, msg, energy=ee)
   if (stat /= ELISHA_OK) then
      call fail(label//': solve E: '//msg) ; return
   endif
   !$acc enter data copyin(buf, pd)
   call dv%solve_packed(mesh, buf, FACTOR, pd, stat, msg, energy=ed, on_device=.true.)
   !$acc update self(pd)
   !$acc exit data delete(buf, pd)
   if (stat /= ELISHA_OK) then
      call fail(label//': solve D: '//msg) ; return
   endif
   call report(label//': E vs ADAM FNL', differ(flat(pe), flat(pa)) + differ([ee], [ea]))
   call report(label//': on_device vs host arrays', differ(flat(pd), flat(pe)) + differ([ed], [ee]))
   call a%destroy ; call e%destroy ; call dv%destroy ; call mesh%destroy
   endsubroutine fft_case

   subroutine facade_case(label, amr)
   !< The facade with both device molds: E (host arrays) against D (on_device), on an AMR or a uniform mesh.
   character(*), intent(in) :: label !< Case label.
   logical,      intent(in) :: amr   !< AMR (hierarchical path), else uniform (spectral path).
   type(grid_object)                           :: grid  !< ADAM's grid (unused).
   type(field_object)                          :: field !< ADAM's field (unused).
   type(elisha_mesh_object)                    :: mesh  !< ELISHA's mesh.
   type(elisha_spectral_solver_cufft_object)   :: smold !< Spectral mold.
   type(elisha_hierarchical_solver_oac_object) :: hmold !< Hierarchical mold.
   type(elisha_solver_object)                  :: e, dv !< Facades, host arrays and device arrays.
   type(elisha_diagnostics_object)             :: de, dd !< Their diagnostics.
   real(R_P), allocatable :: buf(:), pe(:,:,:,:), pd(:,:,:,:) !< Source, potentials.
   real(R_P)              :: ee, ed !< Energies.
   integer(I4P)           :: nb, nbad !< Blocks, differing bits.

   ncase = ncase + 1
   call make_mesh([2,2,2], H16, amr, [0,0,0], .true., grid, field, mesh, buf, nb)
   allocate(pe(1-NG:N3(1)+NG,1-NG:N3(2)+NG,1-NG:N3(3)+NG,1:max(nb,1)))
   allocate(pd, mold=pe)
   pe = 0._R_P ; pd = 0._R_P
   call e%setup(mesh, c, stat, msg, spectral_mold=smold, hierarchical_mold=hmold)
   if (stat == ELISHA_OK) call dv%setup(mesh, c, stat, msg, spectral_mold=smold, hierarchical_mold=hmold)
   if (stat /= ELISHA_OK) then
      call fail(label//': setup: '//msg) ; return
   endif
   call e%solve_packed(mesh, buf, FACTOR, pe, stat, msg, energy=ee)
   if (stat /= ELISHA_OK) then
      call fail(label//': solve E: '//msg) ; return
   endif
   !$acc enter data copyin(buf, pd)
   call dv%solve_packed(mesh, buf, FACTOR, pd, stat, msg, energy=ed, on_device=.true.)
   !$acc update self(pd)
   !$acc exit data delete(buf, pd)
   if (stat /= ELISHA_OK) then
      call fail(label//': solve D: '//msg) ; return
   endif
   de = e%diagnostics() ; dd = dv%diagnostics()
   nbad = differ(flat(pd), flat(pe)) + differ([ed, dd%residual], [ee, de%residual])
   if (dd%iterations /= de%iterations .or. dd%path /= de%path) nbad = nbad + 1
   call report(label//': on_device vs host arrays', nbad)
   call e%destroy ; call dv%destroy ; call mesh%destroy
   endsubroutine facade_case

   pure function flat(p) result(v)
   !< A 4-D array as a vector.
   real(R_P), intent(in)  :: p(:,:,:,:) !< Array.
   real(R_P), allocatable :: v(:)       !< Vector.

   v = reshape(p, [size(p)])
   endfunction flat

   pure function differ(x, y) result(nbad)
   !< The entries of two real arrays whose bit patterns differ.
   real(R_P), intent(in) :: x(:), y(:) !< Arrays.
   integer(I4P)          :: nbad       !< Differing entries.

   nbad = count(transfer(x, [0_int64], size(x)) /= transfer(y, [0_int64], size(y)))
   endfunction differ

   subroutine report(what, nbad)
   !< One comparison, agreed over the ranks: bitwise or not.
   character(*), intent(in) :: what !< The comparison.
   integer(I4P), intent(in) :: nbad !< Differing entries on this rank.
   integer                  :: n(1), ierr_ !< Over the ranks, MPI error status.

   n = nbad
   call MPI_ALLREDUCE(MPI_IN_PLACE, n, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr_)
   if (c%rank == 0) print '(A,A,A,I0,A)', 'elisha_test_device_api_mpi: ', what, ': ', n(1), ' differing'
   if (n(1) /= 0 .and. c%rank == 0) call fail(what//': not bitwise')
   endsubroutine report

   subroutine fail(what)
   !< Report a failed check.
   character(*), intent(in) :: what !< What failed.

   failures = failures + 1
   print '(A,I0,A)', '  FAIL rank ', c%rank, ': '//what
   endsubroutine fail
endprogram elisha_test_device_api_mpi
