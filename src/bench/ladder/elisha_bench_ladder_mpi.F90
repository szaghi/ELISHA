!< ELISHA benchmark: the scaling ladders of the solvers (ADAM's `bench_poisson`, on ELISHA alone), absolute times.
program elisha_bench_ladder_mpi
!< ELISHA benchmark: the scaling ladders of the solvers (ADAM's `bench_poisson`, on ELISHA alone), absolute times.
!<
!< **Why.** ADAM's FLAIL scaling ladders (`src/tests/poisson/bench.sh`, `bench_poisson.F90`; flail.md "Performance")
!< were the only multi-node scaling evidence of these solvers; with ELISHA the source of truth (ELISHA issue #1, EB;
!< ADAM issue #136, D7) they move here, on ELISHA's own program, with absolute times and no ADAM oracle.
!<
!< **The problems (ADAM's).** The octree of the cube [-1,1]^3 (one root block of `ni`^3 cells), a Gaussian source
!< `exp(-|x - (s,0,0)|^2 / 0.05)`; `--path fft`: uniform at `--level` (the spectral solver); `--path mg`: uniform at
!< `--level` with the blocks inside [-1/2,1/2]^3 refined once more (the hierarchical solver; level 2 or more).
!< Strong ladder: level 4, ni 8 (FFT 2 097 152 cells, multigrid 3 932 160); weak: (np, level) = (1, 3), (8, 4),
!< (64, 5). `--bc periodic` (every axis) or `isolated` (free space: the lattice Green's function on the spectral path,
!< the James step on the hierarchical one). Ghost layers `--ngc` (default 3, ADAM's realm: on the isolated spectral
!< path the transform is at least 2 (n + ngc) cells per axis, so it is part of the problem).
!<
!< **Distribution.** `--deal morton` (default): the leaves in Morton (Z) order, the root blocks' octants then each
!< refined block's children, cut into np contiguous chunks of equal leaf counts (ADAM's redistribution keeps blocks
!< along its space-filling curve: neighbours on a rank); `--deal rr`: round-robin, as ELISHA's oracle benches.
!<
!< **What is timed (ADAM's protocol).** The setup once (barrier before, max over the ranks); one untimed warm-up solve
!< (source at s = 0, cold); then `--reps` solves with the source moved to s = 0.01 r (r = 1..reps), the hierarchical
!< solver warm-starting from its last solution as in a time step (`--cold`: every solve cold). Each solve: (device:
!< `!$acc wait`) barrier, `MPI_Wtime`, solve, (device: `!$acc wait`), `MPI_Wtime`, the elapsed time's maximum over the
!< ranks; the minimum, median, mean and maximum over the repetitions. The spectral phase timers (`wall`: gather or
!< redistribution, solve, scatter) per solve as rank 0 measures them, of the spectral solver (`--path fft`) or the
!< hierarchical root (`mg`), and of the James step's ghost-to-ghost solver (`james_*`, isolated `mg`).
!<
!< **Check across rank counts.** `phi_xor` is the XOR over every interior cell of its potential's bits after the
!< warm-up solve, each rotated and keyed by the cell's global index (its leaf's Morton position): independent of the
!< ranks and of the deal, so equal values across np mean bitwise equal potentials (what the slab path promises).
!<
!< **Output.** Rank 0 prints one `BENCH key=value ...` line (machine-readable, the ladder script collects it) and one
!< human line starting with `#`. Not a test (`scripts/run_tests.sh` runs `exe/` only; fpm does not build `src/bench`).
!< Modes: `bench-ladder-gnu`, `bench-ladder-nvf` (host, `exe-bench/`), `bench-ladder-nvf-oac` (cuFFT + OpenACC backends,
!< `exe-bench-oac/`: each rank on the GPU of its node-local rank modulo the node's GPUs; `--on-device` hands the solver
!< device-resident `buf` and `phi`).
!<
!< Usage: `mpirun -np N elisha_bench_ladder_mpi [--path fft|mg] [--bc periodic|isolated] [--level L] [--ni N]
!< [--reps R] [--root-cells C] [--ngc G] [--deal morton|rr] [--cold] [--operator csr|stencil] [--on-device]
!< [--tag LABEL]`; defaults: fft, periodic, level 4, ni 8, reps 5, the backend's root cap, ngc 3, morton, warm.
use mpi
#ifdef _CUFFT
use openacc
use elisha_device, only : elisha_spectral_solver_cufft_object, elisha_hierarchical_solver_oac_object
#endif
use, intrinsic :: ieee_exceptions, only : ieee_usual, ieee_get_halting_mode, ieee_set_halting_mode
use, intrinsic :: iso_fortran_env, only : int64
use elisha, only : I4P, R_P, ELISHA_OK, ELISHA_OPERATOR_CSR, ELISHA_OPERATOR_STENCIL, elisha_comm_object, &
                   elisha_mesh_object, elisha_spectral_solver_object, elisha_hierarchical_solver_object
implicit none
real(R_P), parameter :: FACTOR=4._R_P * acos(-1._R_P) !< Source factor (4 pi G, G = 1).
real(R_P), parameter :: SIGMA2=0.05_R_P               !< Gaussian width squared (ADAM's).
real(R_P), parameter :: SHIFT=0.01_R_P                !< Source shift along x per timed solve (ADAM's).
#ifdef _CUFFT
character(*), parameter :: LEG='fnl' !< The backends: cuFFT + OpenACC.
#else
character(*), parameter :: LEG='cpu' !< The backends: FFTW on the host.
#endif
type(elisha_comm_object)                              :: c      !< Communication context.
type(elisha_mesh_object)                              :: mesh   !< The mesh.
class(elisha_spectral_solver_object),     allocatable :: fft    !< The spectral solver (--path fft).
class(elisha_hierarchical_solver_object), allocatable :: mg     !< The hierarchical solver (--path mg).
character(:), allocatable :: msg, path, bcname, deal, tag, opname !< Message, options.
character(256)            :: arg                                !< Argument.
integer(I4P)              :: level, ni, reps, ngc, root_cells    !< Options.
integer(I4P), allocatable :: rc, lop                             !< Optional setup arguments (present when allocated).
logical                   :: isolated, amr, cold, on_device      !< Options.
integer(I4P), allocatable :: leaves(:,:), c4(:,:), gid(:)        !< Every leaf (Morton order), this rank's, their ids.
real(R_P),    allocatable :: bmin(:,:), buf(:), phi(:,:,:,:)     !< Block corners, source, potential.
real(R_P),    allocatable :: ts(:)                               !< Solve times per repetition.
real(R_P)                 :: t0, t_setup, t_first, w(3), wj(3), residual !< Times, phase walls, residual.
integer(I4P)              :: nleaf, n, lmax, its, its0, stat, a, na !< Counters.
#ifdef _CUFFT
integer                   :: ndev                                !< GPUs per node (max over the nodes).
#endif
integer(int64)            :: hash(1)                             !< phi_xor.
integer                   :: ierr                                !< MPI error status.
logical                   :: halting(3)                          !< The FP halting modes of the build.
character(4)              :: layout                              !< slab, rk0 or red.

call ieee_get_halting_mode(ieee_usual, halting)
call ieee_set_halting_mode(ieee_usual, .false.)
call MPI_INIT(ierr)
call ieee_set_halting_mode(ieee_usual, halting)
call c%init(MPI_COMM_WORLD, stat, msg)
if (stat /= ELISHA_OK) call fail('comm init: '//msg)
! options
path = 'fft' ; bcname = 'periodic' ; deal = 'morton' ; tag = '-' ; opname = 'default'
level = 4 ; ni = 8 ; reps = 5 ; ngc = 3 ; root_cells = 0
cold = .false. ; on_device = .false.
na = command_argument_count()
a = 0
do while (a < na)
   a = a + 1
   call get_command_argument(a, arg)
   select case(trim(arg))
   case('--path')
      path = next_arg()
   case('--bc')
      bcname = next_arg()
   case('--deal')
      deal = next_arg()
   case('--tag')
      tag = next_arg()
   case('--operator')
      opname = next_arg()
   case('--level')
      level = next_int()
   case('--ni')
      ni = next_int()
   case('--reps')
      reps = next_int()
   case('--ngc')
      ngc = next_int()
   case('--root-cells')
      root_cells = next_int()
   case('--cold')
      cold = .true.
   case('--on-device')
      on_device = .true.
   case('--help', '-h')
      if (c%rank == 0) print '(A)', 'usage: elisha_bench_ladder_mpi [--path fft|mg] [--bc periodic|isolated] '// &
         '[--level L] [--ni N] [--reps R] [--root-cells C] [--ngc G] [--deal morton|rr] [--cold] '// &
         '[--operator csr|stencil] [--on-device] [--tag LABEL]'
      call c%destroy(stat)
      call MPI_FINALIZE(ierr)
      stop
   case default
      call fail('unknown argument "'//trim(arg)//'" (--help)')
   endselect
enddo
if (path /= 'fft' .and. path /= 'mg') call fail('--path fft|mg')
if (bcname /= 'periodic' .and. bcname /= 'isolated') call fail('--bc periodic|isolated')
if (deal /= 'morton' .and. deal /= 'rr') call fail('--deal morton|rr')
if (opname /= 'default' .and. opname /= 'csr' .and. opname /= 'stencil') call fail('--operator csr|stencil')
if (level < 0 .or. ni < 1 .or. reps < 1 .or. ngc < 1) call fail('--level >= 0, --ni >= 1, --reps >= 1, --ngc >= 1')
isolated = bcname == 'isolated'
amr = path == 'mg'
if (amr .and. level < 2) call fail('--path mg needs --level 2 or more (the central box refined)')
#ifndef _CUFFT
if (on_device) call fail('--on-device needs the device build (bench-ladder-nvf-oac)')
#else
call select_device
#endif
if (root_cells > 0) allocate(rc, source=root_cells)
if (opname == 'csr') allocate(lop, source=ELISHA_OPERATOR_CSR)
if (opname == 'stencil') allocate(lop, source=ELISHA_OPERATOR_STENCIL)

call build_mesh
call set_source(shift=0._R_P)
allocate(phi(1-ngc:ni+ngc,1-ngc:ni+ngc,1-ngc:ni+ngc,max(n,1)))
phi = 0._R_P
!$acc enter data copyin(buf, phi) if(on_device)

! the setup, timed once
if (amr) then
#ifdef _CUFFT
   allocate(elisha_hierarchical_solver_oac_object :: mg)
#else
   allocate(elisha_hierarchical_solver_object :: mg)
#endif
else
#ifdef _CUFFT
   allocate(elisha_spectral_solver_cufft_object :: fft)
#else
   allocate(elisha_spectral_solver_object :: fft)
#endif
endif
t0 = mark()
if (amr) then
   call mg%setup(mesh, c, stat, msg, isolated=isolated, root_cells=rc, level_operator=lop)
else
   call fft%setup(mesh, c, stat, msg, isolated=isolated)
endif
t_setup = elapsed(t0)
if (stat /= ELISHA_OK) call fail('setup: '//msg)

! the warm-up solve (cold, s = 0), timed apart; its potential hashed
t0 = mark()
call solve
t_first = elapsed(t0)
!$acc update self(phi) if(on_device)
hash = phi_xor()
call MPI_ALLREDUCE(MPI_IN_PLACE, hash, 1, MPI_INTEGER8, MPI_BXOR, MPI_COMM_WORLD, ierr)
call reset_walls

! the timed solves
allocate(ts(reps))
its = 0 ; its0 = 0
do a=1, reps
   call set_source(shift=SHIFT * a)
   !$acc update device(buf) if(on_device)
   if (amr .and. cold) mg%has_guess = .false.
   t0 = mark()
   call solve
   ts(a) = elapsed(t0)
   if (amr) then
      its = its + mg%iterations ; its0 = its0 + mg%iterations0
   else
      its = its + fft%iterations
   endif
enddo
w = 0._R_P ; wj = 0._R_P ; residual = 0._R_P
if (amr) then
   lmax = mg%lmax
   if (allocated(mg%fft)) w = mg%fft%wall / reps
   if (allocated(mg%gfft)) wj = mg%gfft%wall / reps
   layout = 'none'
   if (allocated(mg%fft)) layout = merge('red ', 'slab', mg%fft%redundant)
   residual = mg%residual
else
   w = fft%wall / reps
   layout = merge('slab', 'rk0 ', fft%slab)
endif
if (c%rank == 0) call report
!$acc exit data delete(buf, phi) if(on_device)
if (amr) then
   call mg%destroy
else
   call fft%destroy
endif
call mesh%destroy
call c%destroy(stat)
call MPI_FINALIZE(ierr)

contains
   function next_arg() result(v)
   !< The value of the current option (the next argument).
   character(:), allocatable :: v  !< Value.
   character(256)            :: s  !< Buffer.

   if (a >= na) call fail('option '//trim(arg)//' needs a value')
   a = a + 1
   call get_command_argument(a, s)
   v = trim(s)
   endfunction next_arg

   function next_int() result(v)
   !< The integer value of the current option.
   integer(I4P)              :: v  !< Value.
   character(:), allocatable :: s  !< Text.
   integer                   :: io !< I/O status.

   s = next_arg()
   read(s, *, iostat=io) v
   if (io /= 0) call fail('option '//trim(arg)//' needs an integer, not "'//s//'"')
   endfunction next_int

   subroutine fail(why)
   !< Stop every rank with a message (rank 0 prints it).
   character(*), intent(in) :: why !< Reason.

   if (c%rank == 0) write(*, '(A)') 'elisha_bench_ladder_mpi: '//why
   call MPI_ABORT(MPI_COMM_WORLD, 2, ierr)
   endsubroutine fail

#ifdef _CUFFT
   subroutine select_device
   !< Each rank on the GPU of its node-local rank modulo the node's GPUs, initialised; no GPU stops the run.
   integer :: local_comm, local_rank, nd !< Node communicator, node-local rank, GPUs.

   call MPI_COMM_SPLIT_TYPE(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, local_comm, ierr)
   call MPI_COMM_RANK(local_comm, local_rank, ierr)
   call MPI_COMM_FREE(local_comm, ierr)
   nd = acc_get_num_devices(acc_device_nvidia)
   if (nd < 1) call fail('no NVIDIA GPU visible to this rank')
   call acc_set_device_num(mod(local_rank, nd), acc_device_nvidia)
   call acc_init(acc_device_nvidia)
   call MPI_ALLREDUCE(nd, ndev, 1, MPI_INTEGER, MPI_MAX, MPI_COMM_WORLD, ierr)
   endsubroutine select_device
#endif

   subroutine build_mesh
   !< Every leaf in Morton order (the level-`level` blocks, those inside [-1/2,1/2]^3 replaced by their 8 children when
   !< `amr`), this rank's share (`deal`), the mesh.
   integer(I4P) :: nb, lo, hi, m, i, j, k, o, b, gg, first, last !< Counters.
   logical      :: inbox                                         !< A level-`level` block inside the central box.
   real(R_P)    :: dx0(3), bdx(3)                                !< Level-0 spacing, block size.

   nb = 2_I4P**level ; lo = nb / 4 ; hi = 3 * nb / 4 - 1
   nleaf = 0
   do m=0, nb**3-1
      call morton(m, i, j, k)
      inbox = amr .and. all([i, j, k] >= lo .and. [i, j, k] <= hi)
      nleaf = nleaf + merge(8, 1, inbox)
   enddo
   allocate(leaves(4,nleaf))
   gg = 0
   do m=0, nb**3-1
      call morton(m, i, j, k)
      inbox = amr .and. all([i, j, k] >= lo .and. [i, j, k] <= hi)
      if (inbox) then
         do o=0, 7
            gg = gg + 1
            leaves(:,gg) = [2*i + ibits(o, 0, 1), 2*j + ibits(o, 1, 1), 2*k + ibits(o, 2, 1), level + 1]
         enddo
      else
         gg = gg + 1
         leaves(:,gg) = [i, j, k, level]
      endif
   enddo
   lmax = merge(level + 1, level, amr)
   ! this rank's leaves: a contiguous chunk of the Morton list, or every np-th
   if (deal == 'morton') then
      first = int(int(c%rank, int64) * nleaf / c%nranks, I4P) + 1
      last  = int(int(c%rank + 1, int64) * nleaf / c%nranks, I4P)
      n = max(last - first + 1, 0)
      allocate(gid(n))
      gid = [(gg, gg=first, last)]
   else
      n = 0
      do gg=1, nleaf
         if (mod(gg - 1, c%nranks) == c%rank) n = n + 1
      enddo
      allocate(gid(n))
      b = 0
      do gg=1, nleaf
         if (mod(gg - 1, c%nranks) /= c%rank) cycle
         b = b + 1 ; gid(b) = gg
      enddo
   endif
   dx0 = 2._R_P / ni ! level 0: one block of ni^3 cells on [-1,1]^3
   allocate(c4(4,n), bmin(3,n))
   do b=1, n
      c4(:,b) = leaves(:,gid(b))
      bdx = 2._R_P / 2._R_P**c4(4,b)
      bmin(:,b) = -1._R_P + c4(1:3,b) * bdx
   enddo
   call mesh%init([ni, ni, ni], ngc, [1_I4P, 1_I4P, 1_I4P], dx0, [-1._R_P, -1._R_P, -1._R_P], &
                  spread(.not.isolated, 1, 3), [.false., .false., .false.], c4, stat, msg)
   if (stat /= ELISHA_OK) call fail('mesh init: '//msg)
   call mesh%validate(c, stat, msg)
   if (stat /= ELISHA_OK) call fail('mesh validate: '//msg)
   allocate(buf(max(n*ni**3,1)))
   buf = 0._R_P
   endsubroutine build_mesh

   pure subroutine morton(m, i, j, k)
   !< The block (i, j, k) at Morton position m (bit 3q of m is bit q of i, 3q+1 of j, 3q+2 of k).
   integer(I4P), intent(in)  :: m       !< Morton position.
   integer(I4P), intent(out) :: i, j, k !< Block coordinates.
   integer(I4P)              :: q       !< Bit.

   i = 0 ; j = 0 ; k = 0
   do q=0, 9
      if (btest(m, 3*q))     i = ibset(i, q)
      if (btest(m, 3*q + 1)) j = ibset(j, q)
      if (btest(m, 3*q + 2)) k = ibset(k, q)
   enddo
   endsubroutine morton

   subroutine set_source(shift)
   !< The packed Gaussian source (block-major, i fastest) centred at (shift, 0, 0).
   real(R_P), intent(in) :: shift      !< Centre along x.
   real(R_P)             :: x(3), cdx  !< Cell centre, cell spacing.
   integer(I4P)          :: b, i, j, k, m !< Counters.

   m = 0
   do b=1, n
      cdx = 2._R_P / (ni * 2._R_P**c4(4,b))
      do k=1, ni
         do j=1, ni
            do i=1, ni
               x = bmin(:,b) + ([i, j, k] - 0.5_R_P) * cdx
               x(1) = x(1) - shift
               m = m + 1 ; buf(m) = exp(-sum(x**2) / SIGMA2)
            enddo
         enddo
      enddo
   enddo
   endsubroutine set_source

   subroutine solve
   !< One solve on the chosen path.

   if (amr) then
      call mg%solve_packed(mesh, buf, FACTOR, phi, stat, msg, on_device=on_device)
   else
      call fft%solve_packed(mesh, buf, FACTOR, phi, stat, msg, on_device=on_device)
   endif
   if (stat /= ELISHA_OK) call fail('solve: '//msg)
   endsubroutine solve

   subroutine reset_walls
   !< Zero the spectral phase timers (after the warm-up solve).

   if (amr) then
      if (allocated(mg%fft)) mg%fft%wall = 0._R_P
      if (allocated(mg%gfft)) mg%gfft%wall = 0._R_P
   else
      fft%wall = 0._R_P
   endif
   endsubroutine reset_walls

   function phi_xor() result(h)
   !< This rank's XOR of every interior cell's potential bits, rotated and keyed by the cell's global index.
   integer(int64) :: h            !< Hash.
   integer(int64) :: g, bits      !< Global cell index, potential bits.
   integer(I4P)   :: b, i, j, k   !< Counters.

   h = 0_int64
   do b=1, n
      do k=1, ni
         do j=1, ni
            do i=1, ni
               g = int(gid(b) - 1, int64) * int(ni, int64)**3 + int(((k - 1) * ni + (j - 1)) * ni + i - 1, int64)
               bits = transfer(phi(i,j,k,b), 0_int64)
               h = ieor(h, ieor(ishftc(bits, int(mod(g, 64_int64))), g))
            enddo
         enddo
      enddo
   enddo
   endfunction phi_xor

   function mark() result(t)
   !< The start of a timed region: the device idle, the ranks together, then the clock.
   real(R_P) :: t !< Wall clock.

   !$acc wait
   call MPI_BARRIER(MPI_COMM_WORLD, ierr)
   t = MPI_WTIME()
   endfunction mark

   function elapsed(t) result(tm)
   !< The end of a timed region: the device work complete, then the clock; the maximum over the ranks.
   real(R_P), intent(in) :: t  !< Start.
   real(R_P)             :: tm !< Elapsed, maximum over the ranks.

   !$acc wait
   tm = MPI_WTIME() - t
   call MPI_ALLREDUCE(MPI_IN_PLACE, tm, 1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, ierr)
   endfunction elapsed

   subroutine report
   !< Rank 0: the BENCH line and the human line.
   character(:), allocatable :: line !< The BENCH line.
   integer(int64)            :: cells !< Leaf cells.
   integer(I4P)              :: lbot, root_max !< The hierarchical root level and cap (0 on the spectral path).
   character(16)             :: hx    !< phi_xor in hexadecimal.

   cells = int(nleaf, int64) * int(ni, int64)**3
   lbot = 0 ; root_max = 0
   if (amr) then
      lbot = mg%lbot ; root_max = mg%root_max
   endif
   write(hx, '(Z16.16)') hash(1)
   line = 'BENCH leg='//LEG//' tag='//tag//' path='//path//' bc='//bcname//' amr='//merge('T', 'F', amr)// &
          ' np='//si(c%nranks)//' level='//si(level)//' ni='//si(ni)//' ngc='//si(ngc)//' blocks='//si(nleaf)// &
          ' cells='//si8(cells)//' lmax='//si(lmax)//' lbot='//si(lbot)//' root_max='//si(root_max)// &
          ' layout='//trim(layout)//' deal='//deal//' start='//merge('cold', 'warm', cold .or. .not.amr)// &
          ' operator='//opname//' on_device='//merge('T', 'F', on_device)//' reps='//si(reps)// &
          ' iterations='//si(its)//' iterations0='//si(its0)//' residual='//sr(residual)// &
          ' setup='//sr(t_setup)//' first='//sr(t_first)//' solve_min='//sr(minval(ts))// &
          ' solve_med='//sr(median(ts))//' solve_mean='//sr(sum(ts) / reps)//' solve_max='//sr(maxval(ts))// &
          ' gather='//sr(w(1))//' solve='//sr(w(2))//' scatter='//sr(w(3))// &
          ' james_gather='//sr(wj(1))//' james_solve='//sr(wj(2))//' james_scatter='//sr(wj(3))//' phi_xor='//hx
   print '(A)', line
   print '(A,A,1X,A,A,I0,A,I0,A,I0,A,I0,A,A,A,F10.4,A,F10.5,A,F10.5,A,I0)', '# ', path, bcname, ' np ', c%nranks, &
      ' level ', level, ' ni ', ni, ': ', cells, ' cells, ', trim(layout), ', setup', t_setup, ' s, solve min', &
      minval(ts), ' s, median', median(ts), ' s, iterations ', its
   endsubroutine report

   function si(i) result(s)
   !< An integer as a trimmed string.
   integer(I4P), intent(in)  :: i !< Value.
   character(:), allocatable :: s !< String.
   character(16)             :: w !< Buffer.

   write(w, '(I0)') i
   s = trim(w)
   endfunction si

   function si8(i) result(s)
   !< A 64-bit integer as a trimmed string.
   integer(int64), intent(in) :: i !< Value.
   character(:), allocatable  :: s !< String.
   character(24)              :: w !< Buffer.

   write(w, '(I0)') i
   s = trim(w)
   endfunction si8

   function sr(x) result(s)
   !< A real as a trimmed string (6 significant digits).
   real(R_P), intent(in)     :: x !< Value.
   character(:), allocatable :: s !< String.
   character(24)             :: w !< Buffer.

   write(w, '(ES13.6)') x
   s = trim(adjustl(w))
   endfunction sr

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
endprogram elisha_bench_ladder_mpi
