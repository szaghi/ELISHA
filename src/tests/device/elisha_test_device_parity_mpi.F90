!< ELISHA device test: ELISHA's device backends are bitwise ADAM's device twins, and agree with ELISHA's host solvers.
program elisha_test_device_parity_mpi
!< ELISHA device test: ELISHA's device backends are bitwise ADAM's device twins, and agree with ELISHA's host solvers.
!<
!< Every case runs one source, in one program, through three solvers: ADAM's device twin (`adam_fnl_poisson_fft_object`
!< or `adam_fnl_poisson_mg_object`, ADAM 669d0bdd verbatim, over the shims `adam_grid_object`, `adam_field_object`,
!< `adam_mpih_global`), ELISHA's device backend (`elisha_spectral_solver_cufft_object` or
!< `elisha_hierarchical_solver_oac_object`, over `elisha_mesh_object` and `elisha_comm_object`) and ELISHA's host solver.
!< - **The gate: ELISHA device = ADAM device, bitwise** (one differing bit fails): the potential with every ghost layer
!<   and the energy; for the multigrid, after a first solve and a warm-started second one (the source scaled by 1.25
!<   less 0.1, the boundary values halved), also `iterations`, `iterations0`, `residual`, `tolerance_used`,
!<   `factor_mean`, and once `apply_a` of a fixed vector and `symmetry_defect` of a fixed pair; both must take the same
!<   path (spectral: `slab`, `pcg`, `redundant`; multigrid: `lbot`, `singular`, the root's `redundant` and `slab`).
!< - **ELISHA device against ELISHA host**, within the round-off ADAM's device oracle measured on the A30
!<   (`elisha_test_adam_device_oracle_mpi`: FFT 2.4e-16..5.2e-16, MG `A x` 0, MG potentials 3.7e-16..5.9e-16, equal
!<   iteration counts): `max |phi_dev - phi_host| / max |phi_host|` (every ghost layer) at most `TOL_HOST_FFT` (and the
!<   energies' relative difference at most `TOL_HOST_ENERGY`); for the multigrid `A x` at most `TOL_HOST_AX` and the
!<   potentials at most `TOL_HOST_MG` when the iteration counts agree, `TOL_HOST_FCG_FACTOR` times the stop tolerance
!<   when they do not (printed). Compared only on the same hierarchy (`lbot`): the host's default root is larger than
!<   cuFFT's (`root_cells_max`), so a case with a default root above 32^3 cells is not compared (printed).
!< Cases (blocks dealt round-robin):
!< - spectral (`setup` + `solve_packed`): periodic cubic at level 0, non-cubic at level 1, 2-D (null z); the face kinds
!<   D/D, N/N, D/N and N/D (4n extensions), ghost-Dirichlet, a mix with a periodic axis; isolated; the wide D(G) p = 4;
!< - hierarchical (`setup` + `solve_packed`): periodic uniform (root 64), 2:1 AMR (corner, across the wrap; centre of
!<   3^3, root 216), uniform with one exact V-cycle, ghost-Dirichlet AMR, Dirichlet and mixed Dirichlet/Neumann AMR with
!<   boundary values, Neumann-only AMR with boundary values (singular), isolated uniform and AMR (the James step, its
!<   Green's function on cuFFT's kernel path), 2-D AMR periodic (root 16), and a distributed root (AMR on 5x4x4
!<   level-0 blocks, the root 40x32x32 cells forced by `root_cells`, above the redundant limit: the V-cycle's root on
!<   the slab path, each rank's GPU, the transpose by GPU-aware MPI).
!< Rank 0 prints a line per case and a summary line, `elisha_test_device_parity_mpi: ok (...)` or `FAILED (...)`. Each
!< rank takes the GPU of its node-local rank modulo the node's GPUs; no GPU fails the test. GPU-aware MPI is needed on
!< more than one rank (HPC-X). Built only in `tests-nvf-oac`.
!<
!< `-DELISHA_DEVICE_PARITY_DRYRUN` builds it with the host solvers in place of both device ones (no OpenACC, no cuFFT):
!< a check of the program's own logic on any machine (the gate is then the host parity, and every device-host
!< difference zero). The FP traps of a debug build are off around `MPI_Init` only.
#ifdef ELISHA_DEVICE_PARITY_DRYRUN
#define ADAM_FFT_DEV_T poisson_fft_object
#define ADAM_MG_DEV_T poisson_mg_object
#define ELI_FFT_DEV_T elisha_spectral_solver_object
#define ELI_MG_DEV_T elisha_hierarchical_solver_object
#else
#define ADAM_FFT_DEV_T poisson_fft_fnl_object
#define ADAM_MG_DEV_T poisson_mg_fnl_object
#define ELI_FFT_DEV_T elisha_spectral_solver_cufft_object
#define ELI_MG_DEV_T elisha_hierarchical_solver_oac_object
#endif
use mpi
use, intrinsic :: iso_fortran_env, only : int64
use, intrinsic :: ieee_exceptions, only : ieee_usual, ieee_get_halting_mode, ieee_set_halting_mode
#ifdef _OPENACC
use openacc
#endif
use elisha, only : I4P, R_P, ELISHA_OK, ELISHA_BC_PERIODIC, ELISHA_BC_DIRICHLET, ELISHA_BC_NEUMANN,        &
                   ELISHA_BC_DIRICHLET_GHOST, elisha_comm_object, elisha_mesh_object,                      &
                   elisha_spectral_solver_object, elisha_hierarchical_solver_object, ELISHA_VCYCLE_ADAM
use adam_grid_object, only : grid_object
use adam_field_object, only : field_object
use adam_mpih_global, only : mpih
use adam_poisson_fft_object, only : poisson_fft_object
use adam_poisson_mg_object, only : poisson_mg_object
#ifndef ELISHA_DEVICE_PARITY_DRYRUN
use adam_fnl_poisson_fft_object, only : poisson_fft_fnl_object
use adam_fnl_poisson_mg_object, only : poisson_mg_fnl_object
use elisha_device, only : elisha_spectral_solver_cufft_object, elisha_hierarchical_solver_oac_object
#endif
implicit none
real(R_P),    parameter  :: FACTOR=2.5_R_P              !< Source factor.
real(R_P),    parameter  :: TOL=1.e-11_R_P              !< FCG tolerance.
integer(I4P), parameter  :: MAX_IT=100_I4P              !< FCG iterations allowed.
integer(I4P), parameter  :: NG=2_I4P                    !< Ghost layers (multigrid).
real(R_P),    parameter  :: TOL_HOST_FFT=1.e-14_R_P     !< Spectral: max |phi_dev - phi_host| / max |phi_host|.
real(R_P),    parameter  :: TOL_HOST_ENERGY=1.e-13_R_P  !< Spectral and multigrid: |E_dev - E_host| / |E_host|.
real(R_P),    parameter  :: TOL_HOST_AX=1.e-14_R_P      !< Multigrid: max |A_dev x - A_host x| / max |A_host x|.
real(R_P),    parameter  :: TOL_HOST_MG=1.e-12_R_P      !< Multigrid, equal iterations: potentials' relative difference.
real(R_P),    parameter  :: TOL_HOST_FCG_FACTOR=1.e3_R_P !< Multigrid, other iterations: the bound over the tolerance.
integer(I4P), parameter  :: P=ELISHA_BC_PERIODIC, D=ELISHA_BC_DIRICHLET, NM=ELISHA_BC_NEUMANN !< Kinds.
integer(I4P), parameter  :: G=ELISHA_BC_DIRICHLET_GHOST !< Kind.
logical,      parameter  :: T_=.true., F_=.false.       !< Shorthands.
real(R_P),    parameter  :: DX8(3)=0.125_R_P            !< The spacing of an 8-cell box of side 1.
integer(I4P), parameter  :: N3(3)=[8_I4P, 8_I4P, 8_I4P], N2(3)=[8_I4P, 8_I4P, 1_I4P] !< Cells per block.
real(R_P),    parameter  :: H16(3)=1._R_P / 16._R_P, H24(3)=1._R_P / 24._R_P     !< Spacings of unit boxes.
type(elisha_comm_object) :: c                           !< Communication context.
integer(I4P)             :: fail_gate, fail_host, fail_other, nfail(3), stat, ncase !< Failures by kind; status; cases.
integer                  :: ierr                        !< MPI error status.
integer                  :: ndev                        !< GPUs per node (the most over the ranks).
logical                  :: halting(3)                  !< The FP halting modes of the build (debug: on).
character(:), allocatable :: msg                        !< Message.
real(R_P)                :: worst(4)                    !< Worst dev-host: FFT phi, MG A x, MG phi (equal its), energy.
real(R_P)                :: t_start                     !< Wall clock.

fail_gate = 0 ; fail_host = 0 ; fail_other = 0 ; ncase = 0 ; worst = 0._R_P ; ndev = 0
call ieee_get_halting_mode(ieee_usual, halting)
call ieee_set_halting_mode(ieee_usual, .false.)
call MPI_INIT(ierr)
call ieee_set_halting_mode(ieee_usual, halting)
call mpih%initialize
call c%init(MPI_COMM_WORLD, stat, msg) ; if (stat /= ELISHA_OK) call fail('other', 'comm init: '//msg)
call select_device
t_start = MPI_WTIME()
if (fail_other == 0) then
   if (c%rank == 0) print '(A,I0,A,I0,A)', 'elisha_test_device_parity_mpi: np ', c%nranks, ', GPUs per node ', ndev, &
#ifdef ELISHA_DEVICE_PARITY_DRYRUN
                                           ' (DRY RUN: host solvers in place of the device ones)'
#else
                                           ' (ELISHA device vs ADAM device: bitwise; vs ELISHA host: round-off)'
#endif
   ! spectral
   call fft_case('FFT periodic cubic L0', [4,4,4], [2,2,2], 0, [0.125_R_P, 0.125_R_P, 0.125_R_P], [T_,T_,T_], &
                 [F_,F_,F_], 2)
   call fft_case('FFT periodic non-cubic L1', [4,2,3], [1,2,1], 1, [0.1_R_P, 0.07_R_P, 0.13_R_P], [T_,T_,T_], &
                 [F_,F_,F_], 2)
   call fft_case('FFT periodic 2-D null z', [4,4,1], [2,3,1], 0, [0.05_R_P, 0.08_R_P, 1._R_P], [T_,T_,F_], &
                 [F_,F_,T_], 2)
   call fft_case('FFT faces D/D', [4,4,4], [2,2,2], 0, DX8, [F_,F_,F_], [F_,F_,F_], 2, bc=kinds(D,D, D,D, D,D))
   call fft_case('FFT faces N/N', [4,4,4], [2,2,2], 0, DX8, [F_,F_,F_], [F_,F_,F_], 2, bc=kinds(NM,NM, NM,NM, NM,NM))
   call fft_case('FFT faces D/N (4n)', [4,4,4], [2,2,2], 0, DX8, [F_,F_,F_], [F_,F_,F_], 2, &
                 bc=kinds(D,NM, D,NM, D,NM))
   call fft_case('FFT faces N/D (4n)', [4,4,4], [2,2,2], 0, DX8, [F_,F_,F_], [F_,F_,F_], 2, &
                 bc=kinds(NM,D, NM,D, NM,D))
   call fft_case('FFT faces ghost-D', [4,4,4], [2,2,2], 0, DX8, [F_,F_,F_], [F_,F_,F_], 2, bc=kinds(G,G, G,G, G,G))
   call fft_case('FFT faces P, D/D, N/D', [4,4,4], [2,2,2], 0, [0.1_R_P, 0.07_R_P, 0.13_R_P], [T_,F_,F_], &
                 [F_,F_,F_], 2, bc=kinds(P,P, D,D, NM,D))
   call fft_case('FFT isolated cubic', [4,4,4], [2,2,2], 0, DX8, [F_,F_,F_], [F_,F_,F_], 2, isolated=.true.)
   call fft_case('FFT wide p=4 periodic', [4,4,4], [2,2,2], 0, DX8, [T_,T_,T_], [F_,F_,F_], 3, dg_order=4)
   ! hierarchical
   call mg_case('MG uniform, root 64', [2,2,2], N3, H16, F_, [0,0,0], [F_,F_,F_], T_, F_, 64)
   call mg_case('MG AMR corner (wrap)', [2,2,2], N3, H16, T_, [0,0,0], [F_,F_,F_], T_, F_, 0)
   call mg_case('MG AMR centre, root 216', [3,3,3], N3, H24, T_, [1,1,1], [F_,F_,F_], T_, F_, 216)
   call mg_case('MG uniform, exact root', [2,2,2], N3, H16, F_, [0,0,0], [F_,F_,F_], T_, F_, 0)
   call mg_case('MG AMR ghost-D (default)', [2,2,2], N3, H16, T_, [1,0,1], [F_,F_,F_], F_, F_, 0)
   call mg_case('MG AMR D/D + g', [2,2,2], N3, H16, T_, [1,0,0], [F_,F_,F_], F_, F_, 0, bc=kinds(D,D, D,D, D,D), &
                with_ghost=T_)
   call mg_case('MG AMR D/N,N/D,D/D + g', [2,2,2], N3, [0.1_R_P, 0.07_R_P, 0.13_R_P], T_, [0,1,0], [F_,F_,F_], F_, F_, &
                0, bc=kinds(D,NM, NM,D, D,D), with_ghost=T_)
   call mg_case('MG AMR Neumann (sing.) + g', [2,2,2], N3, H16, T_, [1,1,0], [F_,F_,F_], F_, F_, 0, &
                bc=kinds(NM,NM, NM,NM, NM,NM), with_ghost=T_)
   call mg_case('MG isolated uniform', [2,2,2], N3, H16, F_, [0,0,0], [F_,F_,F_], F_, T_, 0)
   call mg_case('MG isolated AMR', [3,3,3], N3, H24, T_, [1,1,1], [F_,F_,F_], F_, T_, 0)
   call mg_case('MG 2-D AMR periodic, r16', [2,2,1], N2, H16, T_, [0,1,0], [F_,F_,T_], T_, F_, 16)
   call mg_case('MG AMR, slab root 40x32^2', [5,4,4], N3, H16, T_, [2,1,2], [F_,F_,F_], T_, F_, 65536)
endif
nfail = [fail_gate, fail_host, fail_other]
call MPI_ALLREDUCE(MPI_IN_PLACE, nfail, 3, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)
if (c%rank == 0) then
   if (sum(nfail) == 0) then
      print '(A,I0,A,I0,A,I0,A,F7.2,A,4(A,ES9.2))', 'elisha_test_device_parity_mpi: ok (', ncase, ' cases, ', &
         c%nranks, ' rank(s), ', ndev, ' GPU(s) per node, ', MPI_WTIME() - t_start, ' s): ELISHA device = ADAM '// &
         'device bitwise on every case', '; max rel |dev-host|: FFT ', worst(1), ', MG A x ', worst(2), ', MG phi ', &
         worst(3), ', energy ', worst(4)
   else
      print '(A,3(I0,A),I0,A,I0,A,4(A,ES9.2))', 'elisha_test_device_parity_mpi: FAILED (', nfail(1), &
         ' not bitwise ADAM device, ', nfail(2), ' device-host above bound, ', nfail(3), ' other; ', c%nranks, &
         ' rank(s), ', ndev, ' GPU(s) per node)', '; max rel |dev-host|: FFT ', worst(1), ', MG A x ', worst(2), &
         ', MG phi ', worst(3), ', energy ', worst(4)
   endif
endif
call c%destroy(stat)
call MPI_FINALIZE(ierr)
if (sum(nfail) > 0) error stop 1

contains
   subroutine select_device
   !< Each rank on the GPU of its node-local rank modulo the node's GPUs, initialised; no GPU is a failure.
#ifdef _OPENACC
   integer :: local_comm, local_rank, nd, ierr_ !< Node communicator, node-local rank, GPUs, MPI error status.

   call MPI_COMM_SPLIT_TYPE(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, local_comm, ierr_)
   call MPI_COMM_RANK(local_comm, local_rank, ierr_)
   call MPI_COMM_FREE(local_comm, ierr_)
   nd = acc_get_num_devices(acc_device_nvidia)
   if (nd > 0) then
      call acc_set_device_num(mod(local_rank, nd), acc_device_nvidia)
      call acc_init(acc_device_nvidia)
   else
      call fail('other', 'no NVIDIA GPU visible to this rank (the device solvers would not run on the device)')
   endif
   call MPI_ALLREDUCE(nd, ndev, 1, MPI_INTEGER, MPI_MAX, MPI_COMM_WORLD, ierr_)
#else
   ndev = 0
#ifndef ELISHA_DEVICE_PARITY_DRYRUN
   call fail('other', 'built without OpenACC (-acc): the device solvers would run on the host')
#endif
#endif
   endsubroutine select_device

   pure function kinds(lx, hx, ly, hy, lz, hz) result(bc)
   !< The kinds of the six faces as bc(2,3).
   integer(I4P), intent(in) :: lx, hx, ly, hy, lz, hz !< Low and high kinds per axis.
   integer(I4P)             :: bc(2,3)                !< Kinds.

   bc = reshape([lx, hx, ly, hy, lz, hz], [2, 3])
   endfunction kinds

   subroutine fft_case(label, nijk, nblk0, level, dx0, periodic, null, ngc, isolated, bc, dg_order)
   !< One spectral case: ADAM device, ELISHA device and ELISHA host, setup and solve_packed; compare.
   character(*),           intent(in) :: label       !< Case label.
   integer,                intent(in) :: nijk(3)     !< Cells per block.
   integer,                intent(in) :: nblk0(3)    !< Blocks per axis at level 0.
   integer,                intent(in) :: level       !< Level of every block.
   real(R_P),              intent(in) :: dx0(3)      !< Spacing at level 0.
   logical,                intent(in) :: periodic(3) !< Periodic axes.
   logical,                intent(in) :: null(3)     !< Null axes.
   integer,                intent(in) :: ngc         !< Ghost layers.
   logical,      optional, intent(in) :: isolated    !< Isolated boundaries.
   integer(I4P), optional, intent(in) :: bc(2,3)     !< Kinds.
   integer,      optional, intent(in) :: dg_order    !< Operator order.
   real(R_P),    parameter             :: EMIN(3)=[0.3_R_P, -0.2_R_P, 1._R_P] !< Domain corner.
   type(grid_object)                   :: grid        !< ADAM's grid (shim).
   type(field_object)                  :: field       !< ADAM's field (shim).
   type(ADAM_FFT_DEV_T)                :: adam        !< ADAM's device solver.
   type(elisha_mesh_object)            :: mesh        !< ELISHA's mesh.
   type(ELI_FFT_DEV_T)                 :: eli         !< ELISHA's device solver.
   type(elisha_spectral_solver_object) :: hst         !< ELISHA's host solver.
   integer(I4P), allocatable           :: c4(:,:)     !< This rank's blocks [4,nb].
   real(R_P),    allocatable           :: buf(:), pa(:,:,:,:), pe(:,:,:,:), ph(:,:,:,:) !< Source, potentials.
   integer(I4P)                        :: nbl(3), nb, b, g, i, j, k, n, dgo, nbad !< Blocks, counters, order, bits.
   integer(I4P)                        :: gi, gj, gk  !< Global cell.
   real(R_P)                           :: ea, ee, eh, cmp(2), rel, rel_e !< Energies, difference, relative ones.
   logical                             :: same_path   !< ADAM and ELISHA took one path.

   ncase = ncase + 1
   dgo = 0 ; if (present(dg_order)) dgo = dg_order
   nbl = merge(nblk0, nblk0 * 2**level, null)
   nb = 0
   do g=0, product(nbl)-1
      if (mod(g, c%nranks) == c%rank) nb = nb + 1
   enddo
   allocate(c4(4,nb))
   b = 0
   do g=0, product(nbl)-1
      if (mod(g, c%nranks) /= c%rank) cycle
      b = b + 1
      c4(:,b) = [mod(g, nbl(1)), mod(g / nbl(1), nbl(2)), g / (nbl(1) * nbl(2)), int(level, I4P)]
   enddo
   call grid%set(nijk=int(nijk, I4P), ngc=int(ngc, I4P), nblk0=int(nblk0, I4P), dx0=dx0, emin=EMIN, &
                 periodic=periodic, null=null, maxlevel=int(level, I4P))
   call field%set(c4)
   call mesh%init(int(nijk, I4P), int(ngc, I4P), int(nblk0, I4P), dx0, EMIN, periodic, null, c4, stat, msg)
   if (stat /= ELISHA_OK) then
      call fail('other', label//': mesh init: '//msg) ; return
   endif
   ! the packed interior source (block-major, i fastest): smooth, non-periodic, a nonzero mean
   allocate(buf(max(nb * product(nijk), 1)))
   buf = 0._R_P
   n = 0
   do b=1, nb
      do k=1, nijk(3)
         do j=1, nijk(2)
            do i=1, nijk(1)
               gi = c4(1,b) * nijk(1) + i - 1 ; gj = c4(2,b) * nijk(2) + j - 1 ; gk = c4(3,b) * nijk(3) + k - 1
               n = n + 1
               buf(n) = cos(0.7_R_P * gi + 1.3_R_P * gj + 0.4_R_P * gk) + 0.3_R_P * sin(0.21_R_P * gi * gj - 0.5_R_P * gk) &
                        + 0.25_R_P
            enddo
         enddo
      enddo
   enddo
   allocate(pa(1-ngc:nijk(1)+ngc,1-ngc:nijk(2)+ngc,1-ngc:nijk(3)+ngc,max(nb,1)))
   allocate(pe, mold=pa) ; allocate(ph, mold=pa)
   pa = 0._R_P ; pe = 0._R_P ; ph = 0._R_P ; ea = 0._R_P ; ee = 0._R_P ; eh = 0._R_P
   call adam%initialize(grid=grid, field=field, isolated=isolated, bc=bc, dg_order=dgo)
   call adam%solve_packed(field=field, buf=buf, factor=FACTOR, phi=pa, energy=ea)
   call eli%setup(mesh, c, stat, msg, isolated=isolated, bc=bc, dg_order=dgo)
   if (stat /= ELISHA_OK) then
      call fail('other', label//': device setup: '//msg) ; return
   endif
   call eli%solve_packed(mesh, buf, FACTOR, pe, stat, msg, energy=ee)
   if (stat /= ELISHA_OK) then
      call fail('other', label//': device solve: '//msg) ; return
   endif
   call hst%setup(mesh, c, stat, msg, isolated=isolated, bc=bc, dg_order=dgo)
   if (stat /= ELISHA_OK) then
      call fail('other', label//': host setup: '//msg) ; return
   endif
   call hst%solve_packed(mesh, buf, FACTOR, ph, stat, msg, energy=eh)
   if (stat /= ELISHA_OK) then
      call fail('other', label//': host solve: '//msg) ; return
   endif
   ! the gate: bitwise
   nbad = differ(reshape(pa(:,:,:,1:nb), [size(pa(:,:,:,1:nb))]), reshape(pe(:,:,:,1:nb), [size(pe(:,:,:,1:nb))]))
   nbad = nbad + differ([ea], [ee])
   same_path = (adam%slab .eqv. eli%slab) .and. (adam%pcg .eqv. eli%pcg) .and. (adam%redundant .eqv. eli%redundant)
   call MPI_ALLREDUCE(MPI_IN_PLACE, nbad, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)
   ! device against host
   cmp = 0._R_P
   if (nb > 0) cmp = [maxval(abs(pe(:,:,:,1:nb) - ph(:,:,:,1:nb))), maxval(abs(ph(:,:,:,1:nb)))]
   call MPI_ALLREDUCE(MPI_IN_PLACE, cmp, 2, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, ierr)
   rel = cmp(1) / max(cmp(2), tiny(1._R_P))
   rel_e = abs(ee - eh) / max(abs(eh), tiny(1._R_P))
   worst(1) = max(worst(1), rel) ; worst(4) = max(worst(4), rel_e)
   if (c%rank == 0) print '(A,T30,A,A,T42,A,I0,A,ES9.2,A,ES9.2)', label, merge('slab', 'rk0 ', eli%slab), &
      merge(' pcg', '    ', eli%pcg), 'vs ADAM dev: ', nbad, ' bits differ | vs host: phi ', rel, ', energy ', rel_e
   if (nbad /= 0) call fail('gate', label//': ELISHA device is not bitwise ADAM device')
   if (.not.same_path) call fail('gate', label//': ADAM and ELISHA took different paths')
   if (.not.(rel <= TOL_HOST_FFT)) call fail('host', label//': device potential differs from the host one')
   if (.not.(rel_e <= TOL_HOST_ENERGY)) call fail('host', label//': device energy differs from the host one')
   call adam%destroy
   call eli%destroy
   call hst%destroy
   call mesh%destroy
   endsubroutine fft_case

   subroutine mg_case(label, nblk0, nijk, dx0, amr, refine, null, periodic, isolated, root_cells, bc, with_ghost)
   !< One hierarchical case: the same mesh and source through ADAM device, ELISHA device and ELISHA host, two solves
   !< (cold, then warm-started on a changed source and boundary values); compare (see the program).
   character(*), intent(in)           :: label      !< Case label.
   integer,      intent(in)           :: nblk0(3)   !< Blocks per axis at level 0.
   integer(I4P), intent(in)           :: nijk(3)    !< Cells per block per axis.
   real(R_P),    intent(in)           :: dx0(3)     !< Cell spacing at level 0.
   logical,      intent(in)           :: amr        !< Refine block `refine` to level 1.
   integer,      intent(in)           :: refine(3)  !< The refined level-0 block (0-based).
   logical,      intent(in)           :: null(3)    !< Null axes.
   logical,      intent(in)           :: periodic   !< Every active axis periodic (else none).
   logical,      intent(in)           :: isolated   !< Isolated boundaries.
   integer,      intent(in)           :: root_cells !< Most cells of the root (0: each backend's default).
   integer(I4P), intent(in), optional :: bc(2,3)    !< Kinds per face.
   logical,      intent(in), optional :: with_ghost !< Solve with nonzero boundary values.
   real(R_P),    parameter            :: EMIN(3)=[0.3_R_P, -0.2_R_P, 0.1_R_P] !< Domain lower corner.
   type(grid_object)                       :: grid  !< ADAM's grid (shim).
   type(field_object)                      :: field !< ADAM's field (shim).
   type(ADAM_MG_DEV_T)                     :: adam  !< ADAM's device multigrid.
   type(elisha_mesh_object)                :: mesh  !< ELISHA's mesh.
   type(ELI_MG_DEV_T)                      :: eli   !< ELISHA's device hierarchical solver.
   type(elisha_hierarchical_solver_object) :: hst   !< ELISHA's host hierarchical solver.
   integer(I4P), allocatable :: leaves(:,:), coords(:,:)       !< Every leaf, this rank's [4,*].
   real(R_P),    allocatable :: bmin(:,:), bmax(:,:)           !< This rank's block corners [3,nb].
   real(R_P),    allocatable :: q(:,:,:,:,:), buf(:)           !< Source: ADAM's layout and packed.
   real(R_P),    allocatable :: pa(:,:,:,:), pe(:,:,:,:), ph(:,:,:,:) !< Potentials.
   real(R_P),    allocatable :: x(:), y(:), ya(:), ye(:), yh(:), gv(:) !< Fixed vectors, products, boundary values.
   real(R_P)                 :: dx(3), xc(3), len3(3), ea, ee, eh, syma, syme !< Measures.
   real(R_P)                 :: cmp(4), rel_ax, rel_phi, rel_e, bound !< Device-host differences, bound.
   integer(I4P)              :: nleaf, nchild, lmax, nb, b, i, j, k, o, n, ncell, nbad, it(4), ith(2) !< Counters.
   logical                   :: ghost_, same_path, same_tree, same_its !< Boundary values; paths; hierarchy; counts.
   integer                   :: ierr_                        !< MPI error status.

   ncase = ncase + 1
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
      call fail('other', label//': mesh init: '//msg) ; return
   endif
   ! the source, in ADAM's conservative layout and packed (block-major, i fastest)
   ncell = product(nijk)
   allocate(q(1:1,1-NG:nijk(1)+NG,1-NG:nijk(2)+NG,1-NG:nijk(3)+NG,1:max(nb,1)), buf(max(nb*ncell,1)))
   allocate(pa(1-NG:nijk(1)+NG,1-NG:nijk(2)+NG,1-NG:nijk(3)+NG,1:max(nb,1)))
   allocate(pe, mold=pa) ; allocate(ph, mold=pa)
   q = 0._R_P ; buf = 0._R_P ; pa = 0._R_P ; pe = 0._R_P ; ph = 0._R_P
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
   ! the setups: the same arguments for the three solvers
   if (present(bc)) then
      if (root_cells > 0) then
         call adam%initialize(grid=grid, field=field, tolerance=TOL, max_iterations=MAX_IT, bc=bc, root_cells=root_cells)
         call eli%setup(mesh, c, stat, msg, tolerance=TOL, max_iterations=MAX_IT, bc=bc, root_cells=root_cells, &
                        vcycle=ELISHA_VCYCLE_ADAM)
         if (stat == ELISHA_OK) &
            call hst%setup(mesh, c, stat, msg, tolerance=TOL, max_iterations=MAX_IT, bc=bc, root_cells=root_cells, &
                           vcycle=ELISHA_VCYCLE_ADAM)
      else
         call adam%initialize(grid=grid, field=field, tolerance=TOL, max_iterations=MAX_IT, bc=bc)
         call eli%setup(mesh, c, stat, msg, tolerance=TOL, max_iterations=MAX_IT, bc=bc, vcycle=ELISHA_VCYCLE_ADAM)
         if (stat == ELISHA_OK) call hst%setup(mesh, c, stat, msg, tolerance=TOL, max_iterations=MAX_IT, bc=bc, &
                                               vcycle=ELISHA_VCYCLE_ADAM)
      endif
   else
      if (root_cells > 0) then
         call adam%initialize(grid=grid, field=field, tolerance=TOL, max_iterations=MAX_IT, isolated=isolated, &
                              root_cells=root_cells)
         call eli%setup(mesh, c, stat, msg, tolerance=TOL, max_iterations=MAX_IT, isolated=isolated, &
                        root_cells=root_cells, vcycle=ELISHA_VCYCLE_ADAM)
         if (stat == ELISHA_OK) call hst%setup(mesh, c, stat, msg, tolerance=TOL, max_iterations=MAX_IT, &
                                               isolated=isolated, root_cells=root_cells, vcycle=ELISHA_VCYCLE_ADAM)
      else
         call adam%initialize(grid=grid, field=field, tolerance=TOL, max_iterations=MAX_IT, isolated=isolated)
         call eli%setup(mesh, c, stat, msg, tolerance=TOL, max_iterations=MAX_IT, isolated=isolated, &
                        vcycle=ELISHA_VCYCLE_ADAM)
         if (stat == ELISHA_OK) call hst%setup(mesh, c, stat, msg, tolerance=TOL, max_iterations=MAX_IT, &
                                               isolated=isolated, vcycle=ELISHA_VCYCLE_ADAM)
      endif
   endif
   if (stat /= ELISHA_OK) then
      call fail('other', label//': setup: '//msg) ; return
   endif
   ! the boundary values: nonzero on every face, from where each lives
   allocate(gv(max(adam%nbnd,1)))
   gv = 0._R_P
   do i=1, adam%nbnd
      gv(i) = 0.3_R_P + sin(3._R_P * adam%bnd_x(1,i) + 0.4_R_P) * cos(2._R_P * adam%bnd_x(2,i)) + &
              0.2_R_P * adam%bnd_x(3,i)
   enddo
   nbad = 0 ; same_its = .true. ; cmp = 0._R_P ; rel_e = 0._R_P
   ! two solves: the first from zero, the second warm-started from the first on a changed source (and boundary values)
   do o=1, 2
      if (o == 2) then
         q = 1.25_R_P * q - 0.1_R_P ; buf = 1.25_R_P * buf - 0.1_R_P ; gv = 0.5_R_P * gv
      endif
      if (ghost_) then
         call adam%solve(field=field, q=q, iv=1_I4P, factor=FACTOR, phi=pa, energy=ea, ghost=gv)
         call eli%solve_packed(mesh, buf, FACTOR, pe, stat, msg, energy=ee, ghost=gv)
         if (stat == ELISHA_OK) call hst%solve_packed(mesh, buf, FACTOR, ph, stat, msg, energy=eh, ghost=gv)
      else
         call adam%solve(field=field, q=q, iv=1_I4P, factor=FACTOR, phi=pa, energy=ea)
         call eli%solve_packed(mesh, buf, FACTOR, pe, stat, msg, energy=ee)
         if (stat == ELISHA_OK) call hst%solve_packed(mesh, buf, FACTOR, ph, stat, msg, energy=eh)
      endif
      if (stat /= ELISHA_OK) then
         call fail('other', label//': solve: '//msg) ; return
      endif
      nbad = nbad + differ(reshape(pa, [size(pa)]), reshape(pe, [size(pe)]))
      nbad = nbad + differ([ea, adam%residual, adam%tolerance_used, adam%factor_mean], &
                           [ee, eli%residual, eli%tolerance_used, eli%factor_mean])
      if (adam%iterations /= eli%iterations .or. adam%iterations0 /= eli%iterations0) nbad = nbad + 1
      it(2*o-1) = eli%iterations0 ; it(2*o) = eli%iterations ; ith(o) = hst%iterations
      same_its = same_its .and. eli%iterations == hst%iterations .and. eli%iterations0 == hst%iterations0
      if (nb > 0) then
         cmp(3) = max(cmp(3), maxval(abs(pe(:,:,:,1:nb) - ph(:,:,:,1:nb))))
         cmp(4) = max(cmp(4), maxval(abs(ph(:,:,:,1:nb))))
      endif
      rel_e = max(rel_e, abs(ee - eh) / max(abs(eh), tiny(1._R_P)))
   enddo
   ! the operators on fixed vectors
   n = max(nb*ncell,1)
   allocate(x(n), y(n), ya(n), ye(n), yh(n))
   do i=1, n
      x(i) = sin(1.3_R_P * i + 0.7_R_P * c%rank) ; y(i) = cos(2.1_R_P * i - 0.4_R_P * c%rank)
   enddo
   syma = adam%symmetry_defect(x, y)
   syme = eli%symmetry_defect(x, y, stat)
   nbad = nbad + differ([syma], [syme])
   ya = 0._R_P ; ye = 0._R_P ; yh = 0._R_P
   call adam%apply_a(x, ya)
   call eli%apply_a(x, ye, stat)
   call hst%apply_a(x, yh, stat)
   nbad = nbad + differ(ya, ye)
   if (nb > 0) cmp(1:2) = [maxval(abs(ye(1:nb*ncell) - yh(1:nb*ncell))), maxval(abs(yh(1:nb*ncell)))]
   ! the path
   same_path = adam%lbot == eli%lbot .and. (adam%singular .eqv. eli%singular) .and. &
               (adam%fft%redundant .eqv. eli%fft%redundant) .and. (adam%fft%slab .eqv. eli%fft%slab)
   same_tree = eli%lbot == hst%lbot .and. (eli%fft%redundant .eqv. hst%fft%redundant)
   call MPI_ALLREDUCE(MPI_IN_PLACE, nbad, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr_)
   call MPI_ALLREDUCE(MPI_IN_PLACE, cmp, 4, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, ierr_)
   rel_ax = cmp(1) / max(cmp(2), tiny(1._R_P)) ; rel_phi = cmp(3) / max(cmp(4), tiny(1._R_P))
   bound = merge(TOL_HOST_MG, TOL_HOST_FCG_FACTOR * max(eli%tolerance_used, hst%tolerance_used), same_its)
   if (c%rank == 0) then
      print '(A,T30,A,I0,A,A,A,I0,A,I0,A,I0,A,I0,A,I0,A)', label, 'lbot ', eli%lbot, ', ', &
         merge('redundant', 'slab     ', eli%fft%redundant), ', its ', it(1), '+', it(2), ', ', it(3), '+', it(4), &
         ' (host ', ith(2), ')'
      if (same_tree) then
         print '(T30,A,I0,A,ES9.2,A,ES9.2,A,ES9.2,A,ES9.2,A)', 'vs ADAM dev: ', nbad, ' differ | vs host: A x ', &
            rel_ax, ', phi ', rel_phi, ' (bound ', bound, '), energy ', rel_e, merge('                ', &
            ' its differ     ', same_its)
      else
         print '(T30,A,I0,A,I0,A)', 'vs ADAM dev: ', nbad, ' differ | vs host: not compared (host lbot ', hst%lbot, &
            ', the backends'' default roots differ)'
      endif
   endif
   if (nbad /= 0) call fail('gate', label//': ELISHA device is not bitwise ADAM device')
   if (.not.same_path) call fail('gate', label//': ADAM and ELISHA took different paths')
   if (same_tree) then
      worst(2) = max(worst(2), rel_ax) ; worst(4) = max(worst(4), rel_e)
      if (same_its) worst(3) = max(worst(3), rel_phi)
      if (.not.(rel_ax <= TOL_HOST_AX)) call fail('host', label//': device operator differs from the host one')
      if (.not.(rel_phi <= bound)) call fail('host', label//': device potential differs from the host one')
      if (.not.(rel_e <= TOL_HOST_ENERGY .or. .not.same_its)) &
         call fail('host', label//': device energy differs from the host one')
   endif
   call adam%destroy
   call eli%destroy
   call hst%destroy
   call mesh%destroy
   endsubroutine mg_case

   pure function differ(a, e) result(nbad)
   !< The entries of two real arrays whose bit patterns differ.
   real(R_P), intent(in) :: a(:), e(:) !< Arrays.
   integer(I4P)          :: nbad       !< Differing entries.

   nbad = count(transfer(a, [0_int64], size(a)) /= transfer(e, [0_int64], size(e)))
   endfunction differ

   subroutine fail(kind, what)
   !< Report a failed check of a kind: 'gate' (not bitwise ADAM device), 'host' (device-host), 'other'.
   character(*), intent(in) :: kind !< Kind.
   character(*), intent(in) :: what !< What failed.

   select case(kind)
   case('gate') ; fail_gate = fail_gate + 1
   case('host') ; fail_host = fail_host + 1
   case default ; fail_other = fail_other + 1
   endselect
   print '(A,I0,A)', '  FAIL ('//kind//') rank ', c%rank, ': '//what
   endsubroutine fail
endprogram elisha_test_device_parity_mpi
