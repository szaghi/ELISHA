!< ELISHA benchmark: the device backends' solve times against ADAM's device twins (OpenACC + cuFFT), same inputs.
program elisha_bench_device_mpi
!< ELISHA benchmark: the device backends' solve times against ADAM's device twins (OpenACC + cuFFT), same inputs.
!<
!< ELISHA's multigrid runs ADAM's V-cycle (`vcycle=ELISHA_VCYCLE_ADAM`): the same algorithm, compared bitwise.
!< ADAM's FLAIL "Performance" problem on the GPU: the cube [-1,1]^3 with a Gaussian source. ADAM's solvers are the
!< verbatim device twins (`adam_fnl_poisson_fft_object`, `adam_fnl_poisson_mg_object`, over the shims), ELISHA's the
!< device backends (`elisha_spectral_solver_cufft_object`, `elisha_hierarchical_solver_oac_object`, the root on cuFFT,
!< its default cap 32^3 cells). Cases (letters):
!< - `a` spectral 128^3 periodic, `b` spectral 128^3 isolated (4^3 blocks of 32^3, the host bench's), `c` spectral
!<   64^3 periodic (4^3 blocks of 16^3);
!< - `d` hierarchical AMR periodic, `e` hierarchical AMR isolated (the James step: two FCG solves): blocks of 8^3
!<   cells, uniform at level 3 (64^3 cells) with the central box [-1/2,1/2]^3 refined once, 960 blocks, 491 520 cells
!<   (the host bench's cases (a) and (c));
!< - `f`, `g`: the same at level 4 with the central box refined, 7 680 blocks, 3 932 160 cells: ADAM's FNL table
!<   (flail.md, "Performance": one A30, 0.33 s periodic, 0.85 s isolated). Not in the default list: ADAM's host-side
!<   setup of it is long, and much longer above one rank (its table, the audit's key mixing).
!< The blocks (leaves) are dealt round-robin. Both solvers get the same packed source; on the untimed first pair
!< their potentials (every ghost layer), and for the multigrid the iterations and the residual, must be bitwise
!< equal. Every multigrid solve is a cold one (`has_guess` cleared). Timing: wall clock (`MPI_Wtime`); before every
!< solve `!$acc wait` and a barrier, after it `!$acc wait` before the clock is read (the solve's device work is
!< complete: the potential comes back to the host synchronously anyway, the wait guards any queued work); ADAM and
!< ELISHA solves alternated (A, E, A, E, ...) after the untimed pair, each solve's time the maximum over the ranks;
!< the minimum and the median over the repetitions; the setups (host assembly, plans, device placement) timed once
!< each, apart. Each rank takes the GPU of its node-local rank modulo the node's GPUs (np 2 on one A30 shares it);
!< no GPU stops the run. GPU-aware MPI is needed on more than one rank. Not a test (`scripts/run_tests.sh` runs `exe/`
!< only; fpm does not build `src/bench`). Mode `bench-nvf-oac` (`exe-bench-oac/`). Usage:
!< `mpirun -np N exe-bench-oac/elisha_bench_device_mpi [reps [AE|EA [cases]]]`: `reps` the timed pairs (default 9),
!< `EA` times ELISHA first in every pair (a control for an order bias; `AE`, the default, ADAM first), `cases` the
!< letters of the cases (default `abcde`); `stencil` as a fourth argument sets the multigrid up with
!< `level_operator=ELISHA_OPERATOR_STENCIL` (M7, audit C2; default `csr`).
use mpi
use openacc
use, intrinsic :: ieee_exceptions, only : ieee_usual, ieee_get_halting_mode, ieee_set_halting_mode
use, intrinsic :: iso_fortran_env, only : int64
use elisha, only : I4P, R_P, ELISHA_OK, ELISHA_OPERATOR_CSR, ELISHA_OPERATOR_STENCIL, ELISHA_VCYCLE_ADAM, &
                   elisha_comm_object, elisha_mesh_object
use elisha_device, only : elisha_spectral_solver_cufft_object, elisha_hierarchical_solver_oac_object
use adam_grid_object, only : grid_object
use adam_field_object, only : field_object
use adam_mpih_global, only : mpih
use adam_fnl_poisson_fft_object, only : poisson_fft_fnl_object
use adam_fnl_poisson_mg_object, only : poisson_mg_fnl_object
implicit none
real(R_P), parameter     :: FACTOR=4._R_P * acos(-1._R_P) !< Source factor (4 pi G, G = 1).
type(elisha_comm_object) :: c          !< Communication context.
integer(I4P)             :: stat, reps !< Status, repetitions.
integer                  :: ierr, na   !< MPI error status, arguments.
integer                  :: ndev       !< GPUs per node.
logical                  :: halting(3) !< The FP halting modes of the build.
logical                  :: efirst=.false. !< ELISHA first in every pair.
character(:), allocatable :: msg       !< Message.
character(16)            :: arg        !< Argument.
character(16)            :: cases='abcde' !< The cases to run.
integer(I4P)             :: lop=ELISHA_OPERATOR_CSR !< ELISHA's level operators (the multigrid).

call ieee_get_halting_mode(ieee_usual, halting)
call ieee_set_halting_mode(ieee_usual, .false.)
call MPI_INIT(ierr)
call ieee_set_halting_mode(ieee_usual, halting)
call mpih%initialize
call c%init(MPI_COMM_WORLD, stat, msg)
if (stat /= ELISHA_OK) error stop 'comm init'
call select_device
reps = 9
na = command_argument_count()
if (na > 0) then
   call get_command_argument(1, arg)
   read(arg, *) reps
endif
if (na > 1) then
   call get_command_argument(2, arg)
   efirst = trim(arg) == 'EA'
endif
if (na > 2) call get_command_argument(3, cases)
if (na > 3) then
   call get_command_argument(4, arg)
   if (trim(arg) == 'stencil') lop = ELISHA_OPERATOR_STENCIL
endif
if (c%rank == 0) then
   print '(A,I0,A,I0,A,I0,A,A,A,A)', '# elisha_bench_device_mpi: ', c%nranks, ' rank(s), ', ndev, ' GPU(s) per node, ', &
         reps, ' repetitions (max over ranks), order ', merge('E, A', 'A, E', efirst), ', ELISHA level operators ', &
         trim(merge('stencil', 'csr    ', lop == ELISHA_OPERATOR_STENCIL))
   print '(A)', '# case                      cells   lv  it path setup_A[s] setup_E[s] solve_A[s] solve_E[s] E/A(min) '// &
                ' med_A[s]   med_E[s]  E/A(med) bitwise'
endif
if (index(cases, 'a') > 0) call fft_case('(a) FFT 128^3 periodic ', 32, .false.)
if (index(cases, 'b') > 0) call fft_case('(b) FFT 128^3 isolated ', 32, .true.)
if (index(cases, 'c') > 0) call fft_case('(c) FFT  64^3 periodic ', 16, .false.)
if (index(cases, 'd') > 0) call mg_case('(d) MG AMR periodic, L3', 3_I4P, periodic=.true., isolated=.false.)
if (index(cases, 'e') > 0) call mg_case('(e) MG AMR isolated, L3', 3_I4P, periodic=.false., isolated=.true.)
if (index(cases, 'f') > 0) call mg_case('(f) MG AMR periodic, L4', 4_I4P, periodic=.true., isolated=.false.)
if (index(cases, 'g') > 0) call mg_case('(g) MG AMR isolated, L4', 4_I4P, periodic=.false., isolated=.true.)
call c%destroy(stat)
call MPI_FINALIZE(ierr)

contains
   subroutine select_device
   !< Each rank on the GPU of its node-local rank modulo the node's GPUs, initialised; no GPU stops the run.
   integer :: local_comm, local_rank, nd !< Node communicator, node-local rank, GPUs.

   call MPI_COMM_SPLIT_TYPE(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, local_comm, ierr)
   call MPI_COMM_RANK(local_comm, local_rank, ierr)
   call MPI_COMM_FREE(local_comm, ierr)
   nd = acc_get_num_devices(acc_device_nvidia)
   if (nd < 1) error stop 'elisha_bench_device_mpi: no NVIDIA GPU visible to this rank'
   call acc_set_device_num(mod(local_rank, nd), acc_device_nvidia)
   call acc_init(acc_device_nvidia)
   call MPI_ALLREDUCE(nd, ndev, 1, MPI_INTEGER, MPI_MAX, MPI_COMM_WORLD, ierr)
   endsubroutine select_device

   function mark() result(t0)
   !< The start of a timed region: the device idle, the ranks together, then the clock.
   real(R_P) :: t0 !< Wall clock.

   !$acc wait
   call MPI_BARRIER(MPI_COMM_WORLD, ierr)
   t0 = MPI_WTIME()
   endfunction mark

   function elapsed(t0) result(tm)
   !< The end of a timed region: the device work complete, then the clock; the maximum over the ranks.
   real(R_P), intent(in) :: t0 !< Start.
   real(R_P)             :: tm !< Elapsed, maximum over the ranks.

   !$acc wait
   tm = MPI_WTIME() - t0
   call MPI_ALLREDUCE(MPI_IN_PLACE, tm, 1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, ierr)
   endfunction elapsed

   subroutine fft_case(label, nc, isolated)
   !< One spectral case: 4^3 blocks of nc^3 cells on [-1,1]^3, periodic or isolated; both setups, then the alternated
   !< solves.
   character(*), intent(in) :: label    !< Case label.
   integer,      intent(in) :: nc       !< Cells per block per axis.
   logical,      intent(in) :: isolated !< Isolated boundaries.
   integer(I4P), parameter  :: NGC=2_I4P, NB0=4_I4P !< Ghost layers, blocks per axis.
   real(R_P),    parameter  :: SIGMA=0.1_R_P        !< Gaussian width.
   type(grid_object)                         :: grid  !< ADAM's grid (shim).
   type(field_object)                        :: field !< ADAM's field (shim).
   type(poisson_fft_fnl_object)              :: adam  !< ADAM's device solver.
   type(elisha_mesh_object)                  :: mesh  !< ELISHA's mesh.
   type(elisha_spectral_solver_cufft_object) :: eli   !< ELISHA's device solver.
   integer(I4P), allocatable :: c4(:,:)          !< This rank's blocks.
   real(R_P),    allocatable :: buf(:), pa(:,:,:,:), pe(:,:,:,:) !< Source, potentials.
   real(R_P),    allocatable :: ta(:), te(:)     !< Solve times per repetition.
   real(R_P)                 :: dx, x(3), sa, se, t0 !< Spacing, cell centre, setup times, mark.
   integer(I4P)              :: n, b, g, i, j, k, m, r, slot, nbad(1) !< Counters, mismatches.

   dx = 2._R_P / (NB0 * nc)
   n = 0
   do g=0, NB0**3-1
      if (mod(g, c%nranks) == c%rank) n = n + 1
   enddo
   allocate(c4(4,n))
   b = 0
   do g=0, NB0**3-1
      if (mod(g, c%nranks) /= c%rank) cycle
      b = b + 1 ; c4(:,b) = [mod(g, NB0), mod(g / NB0, NB0), g / NB0**2, 0_I4P]
   enddo
   call grid%set(nijk=[nc, nc, nc], ngc=NGC, nblk0=[NB0, NB0, NB0], dx0=[dx, dx, dx], emin=[-1._R_P, -1._R_P, -1._R_P], &
                 periodic=spread(.not.isolated, 1, 3), null=[.false., .false., .false.], maxlevel=0_I4P)
   call field%set(c4)
   call mesh%init([nc, nc, nc], NGC, [NB0, NB0, NB0], [dx, dx, dx], [-1._R_P, -1._R_P, -1._R_P], &
                  spread(.not.isolated, 1, 3), [.false., .false., .false.], c4, stat, msg)
   if (stat /= ELISHA_OK) error stop 'mesh init'
   allocate(buf(max(n*nc**3,1)))
   m = 0
   do b=1, n
      do k=1, nc
         do j=1, nc
            do i=1, nc
               x = -1._R_P + ([c4(1,b) * nc + i, c4(2,b) * nc + j, c4(3,b) * nc + k] - 0.5_R_P) * dx
               m = m + 1 ; buf(m) = exp(-sum(x**2) / (2._R_P * SIGMA**2))
            enddo
         enddo
      enddo
   enddo
   allocate(pa(1-NGC:nc+NGC,1-NGC:nc+NGC,1-NGC:nc+NGC,max(n,1)), pe(1-NGC:nc+NGC,1-NGC:nc+NGC,1-NGC:nc+NGC,max(n,1)))
   pa = 0._R_P ; pe = 0._R_P
   ! the setups, once each
   t0 = mark()
   call adam%initialize(grid=grid, field=field, isolated=isolated)
   sa = elapsed(t0)
   t0 = mark()
   call eli%setup(mesh, c, stat, msg, isolated=isolated)
   se = elapsed(t0)
   if (stat /= ELISHA_OK) error stop 'setup'
   ! one untimed pair, checked bitwise, then the alternated timed solves
   call adam%solve_packed(field=field, buf=buf, factor=FACTOR, phi=pa)
   call eli%solve_packed(mesh, buf, FACTOR, pe, stat)
   nbad = count(transfer(pa(:,:,:,1:n), [0_int64]) /= transfer(pe(:,:,:,1:n), [0_int64]))
   call MPI_ALLREDUCE(MPI_IN_PLACE, nbad, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)
   allocate(ta(reps), te(reps))
   do r=1, reps
      do slot=1, 2
         t0 = mark()
         if ((slot == 1) .neqv. efirst) then
            call adam%solve_packed(field=field, buf=buf, factor=FACTOR, phi=pa)
            ta(r) = elapsed(t0)
         else
            call eli%solve_packed(mesh, buf, FACTOR, pe, stat)
            te(r) = elapsed(t0)
         endif
      enddo
   enddo
   if (c%rank == 0) print '(A,I10,7X,A4,2F11.4,2F11.5,F9.4,2F11.5,F9.4,2X,A)', label, (NB0 * nc)**3, &
      merge('slab', 'rk0 ', eli%slab), sa, se, minval(ta), minval(te), minval(te) / minval(ta), median(ta), &
      median(te), median(te) / median(ta), merge('yes', 'NO ', nbad(1) == 0)
   call adam%destroy
   call eli%destroy
   call mesh%destroy
   endsubroutine fft_case

   subroutine mg_case(label, l0, periodic, isolated)
   !< One hierarchical case: the level-l0 octree of [-1,1]^3 (2^l0 blocks per axis of 8^3 cells), the central box
   !< refined once; both setups (the cuFFT root by default), then the alternated cold solves.
   character(*), intent(in) :: label    !< Case label.
   integer(I4P), intent(in) :: l0       !< Level of the uniform blocks.
   logical,      intent(in) :: periodic !< Periodic axes (all or none).
   logical,      intent(in) :: isolated !< Isolated boundaries.
   integer(I4P), parameter  :: NC=8_I4P, NGC=2_I4P !< Cells per block, ghost layers.
   real(R_P),    parameter  :: SIGMA2=0.05_R_P     !< Gaussian width squared (ADAM's).
   type(grid_object)                           :: grid  !< ADAM's grid (shim).
   type(field_object)                          :: field !< ADAM's field (shim).
   type(poisson_mg_fnl_object)                 :: adam  !< ADAM's device multigrid.
   type(elisha_mesh_object)                    :: mesh  !< ELISHA's mesh.
   type(elisha_hierarchical_solver_oac_object) :: eli   !< ELISHA's device hierarchical solver.
   integer(I4P), allocatable :: leaves(:,:), c4(:,:)      !< Every leaf, this rank's.
   real(R_P),    allocatable :: bmin(:,:), bmax(:,:)      !< This rank's block corners.
   real(R_P),    allocatable :: buf(:), pa(:,:,:,:), pe(:,:,:,:) !< Source, potentials.
   real(R_P),    allocatable :: ta(:), te(:)              !< Solve times per repetition.
   real(R_P)                 :: dx(3), x(3), sa, se, t0   !< Spacing, cell centre, setup times, mark.
   integer(I4P)              :: nb, lo, hi, nleaf, n, b, gg, i, j, k, m, o, r, slot, nbad(1), ita, ite !< Counters.
   logical                   :: inbox                     !< A level-l0 block inside the central box.

   ! every leaf: level-l0 blocks i fastest, the ones inside [-1/2,1/2]^3 replaced by their 8 children
   nb = 2_I4P**l0 ; lo = nb / 4 ; hi = 3 * nb / 4 - 1
   nleaf = 0
   do k=0, nb-1
      do j=0, nb-1
         do i=0, nb-1
            inbox = all([i, j, k] >= lo .and. [i, j, k] <= hi)
            nleaf = nleaf + merge(8, 1, inbox)
         enddo
      enddo
   enddo
   allocate(leaves(4,nleaf))
   gg = 0
   do k=0, nb-1
      do j=0, nb-1
         do i=0, nb-1
            inbox = all([i, j, k] >= lo .and. [i, j, k] <= hi)
            if (inbox) then
               do o=0, 7
                  gg = gg + 1
                  leaves(:,gg) = [2*i + ibits(o, 0, 1), 2*j + ibits(o, 1, 1), 2*k + ibits(o, 2, 1), l0 + 1]
               enddo
            else
               gg = gg + 1
               leaves(:,gg) = [i, j, k, l0]
            endif
         enddo
      enddo
   enddo
   dx = 2._R_P / NC ! level 0: one block of NC^3 cells on [-1,1]^3
   call grid%set(nijk=[NC, NC, NC], ngc=NGC, nblk0=[1_I4P, 1_I4P, 1_I4P], dx0=dx, emin=[-1._R_P, -1._R_P, -1._R_P], &
                 periodic=spread(periodic, 1, 3), null=[.false., .false., .false.], maxlevel=l0 + 1)
   n = 0
   do gg=0, nleaf-1
      if (mod(gg, c%nranks) == c%rank) n = n + 1
   enddo
   allocate(c4(4,n), bmin(3,n), bmax(3,n))
   b = 0
   do gg=0, nleaf-1
      if (mod(gg, c%nranks) /= c%rank) cycle
      b = b + 1 ; c4(:,b) = leaves(:,gg+1)
      bmin(:,b) = grid%domain_emin + c4(1:3,b) * grid%block_dxyz(:,c4(4,b))
      bmax(:,b) = bmin(:,b) + grid%block_dxyz(:,c4(4,b))
   enddo
   call field%set(c4, emin=bmin, emax=bmax)
   call mesh%init([NC, NC, NC], NGC, [1_I4P, 1_I4P, 1_I4P], dx, [-1._R_P, -1._R_P, -1._R_P], spread(periodic, 1, 3), &
                  [.false., .false., .false.], c4, stat, msg)
   if (stat /= ELISHA_OK) error stop 'mesh init'
   allocate(buf(max(n*NC**3,1)))
   m = 0
   do b=1, n
      do k=1, NC
         do j=1, NC
            do i=1, NC
               x = bmin(:,b) + ([i, j, k] - 0.5_R_P) * grid%cell_dxyz(:,c4(4,b))
               m = m + 1 ; buf(m) = exp(-sum(x**2) / SIGMA2)
            enddo
         enddo
      enddo
   enddo
   allocate(pa(1-NGC:NC+NGC,1-NGC:NC+NGC,1-NGC:NC+NGC,max(n,1)), pe(1-NGC:NC+NGC,1-NGC:NC+NGC,1-NGC:NC+NGC,max(n,1)))
   pa = 0._R_P ; pe = 0._R_P
   ! the setups, once each (host assembly, the root's plans, the device placement)
   t0 = mark()
   call adam%initialize(grid=grid, field=field, isolated=isolated)
   sa = elapsed(t0)
   t0 = mark()
   call eli%setup(mesh, c, stat, msg, isolated=isolated, level_operator=lop, vcycle=ELISHA_VCYCLE_ADAM)
   se = elapsed(t0)
   if (stat /= ELISHA_OK) error stop 'setup'
   ! one untimed pair (cold), checked bitwise, then the alternated timed cold solves
   adam%has_guess = .false. ; eli%has_guess = .false.
   call adam%solve_packed(field=field, buf=buf, factor=FACTOR, phi=pa)
   call eli%solve_packed(mesh, buf, FACTOR, pe, stat)
   nbad = count(transfer(pa(:,:,:,1:n), [0_int64]) /= transfer(pe(:,:,:,1:n), [0_int64]))
   ita = adam%iterations0 + adam%iterations ; ite = eli%iterations0 + eli%iterations
   if (ita /= ite .or. transfer(adam%residual, 0_int64) /= transfer(eli%residual, 0_int64)) nbad = nbad + 1
   call MPI_ALLREDUCE(MPI_IN_PLACE, nbad, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)
   allocate(ta(reps), te(reps))
   do r=1, reps
      do slot=1, 2
         adam%has_guess = .false. ; eli%has_guess = .false.
         t0 = mark()
         if ((slot == 1) .neqv. efirst) then
            call adam%solve_packed(field=field, buf=buf, factor=FACTOR, phi=pa)
            ta(r) = elapsed(t0)
         else
            call eli%solve_packed(mesh, buf, FACTOR, pe, stat)
            te(r) = elapsed(t0)
         endif
      enddo
   enddo
   if (c%rank == 0) print '(A,I10,1X,I2,A,I2,I4,1X,A4,2F11.4,2F11.5,F9.4,2F11.5,F9.4,2X,A)', label, mesh%nleaf * NC**3, &
      eli%lbot, '..', eli%lmax, ite, merge('red ', 'slab', eli%fft%redundant), sa, se, minval(ta), minval(te), &
      minval(te) / minval(ta), median(ta), median(te), median(te) / median(ta), merge('yes', 'NO ', nbad(1) == 0)
   call adam%destroy
   call eli%destroy
   call mesh%destroy
   endsubroutine mg_case

   pure function median(t) result(md)
   !< The median of t (insertion sort of a copy).
   real(R_P), intent(in) :: t(:) !< Values.
   real(R_P)             :: md   !< Median.
   real(R_P)             :: s(size(t)), v !< Sorted copy, value.
   integer               :: i, j !< Counters.

   s = t
   do i=2, size(s)
      v = s(i) ; j = i - 1
      do while (j >= 1)
         if (s(j) <= v) exit
         s(j+1) = s(j) ; j = j - 1
      enddo
      s(j+1) = v
   enddo
   if (mod(size(s), 2) == 1) then
      md = s((size(s) + 1) / 2)
   else
      md = 0.5_R_P * (s(size(s) / 2) + s(size(s) / 2 + 1))
   endif
   endfunction median
endprogram elisha_bench_device_mpi
