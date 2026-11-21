!< ELISHA benchmark: the hierarchical solver's solve time against ADAM's multigrid (the verbatim oracle), same inputs.
program elisha_bench_hierarchical_mpi
!< ELISHA benchmark: the hierarchical solver's solve time against ADAM's multigrid (the verbatim oracle), same inputs.
!<
!< ADAM's FLAIL "Performance" problem scaled down: the cube [-1,1]^3 with a Gaussian source (exp(-r^2 / 0.05)), blocks
!< of 8^3 cells, uniform at level 3 (8^3 blocks, 64^3 cells) with the central box [-1/2,1/2]^3 refined once (level 4):
!< 960 blocks, 491 520 cells. Cases: (a) periodic; (b) ghost-Dirichlet faces (bounded); (c) isolated (the James step:
!< two FCG solves); (d) uniform 64^3, face kinds D/N, N/D, D/D, the root capped at 32^3 cells, so the V-cycle smooths
!< two levels and its root is the redundant path with an extended transform (128 x 128 x 64 cells on every rank). The
!< leaves are dealt round-robin. Both solvers get the same packed source, ELISHA's with ADAM's V-cycle
!< (`vcycle=ELISHA_VCYCLE_ADAM`: the same algorithm); their potentials and iterations must be bitwise equal (checked on
!< the untimed pair). Every timed solve is a cold one (`has_guess` cleared: the warm start
!< would make a repeated solve take no iteration). Timing: wall clock (`MPI_Wtime`), a barrier before every solve, ADAM
!< and ELISHA solves alternated (A, E, A, E, ...) after one untimed pair, each solve's time the maximum over the ranks;
!< the minimum and the median over the repetitions; the setups timed once each, apart. Output: one row per case. Not a
!< test (`scripts/run_tests.sh` runs `exe/` only; fpm does not build `src/bench`). Modes `bench-hierarchical-gnu`,
!< `bench-hierarchical-nvf`. Usage: `mpirun -np N exe-bench/elisha_bench_hierarchical_mpi [reps [EA|AE [cases]]]`:
!< `EA` times ELISHA first in every pair (a control for an order bias), the default is ADAM first; `cases` the letters
!< of the cases to run (default `abcd`: the AMR setups at np > 1 take tens of seconds, so reruns pick their cases);
!< `stencil` as a fourth argument sets ELISHA up with `level_operator=ELISHA_OPERATOR_STENCIL` (M7, audit C2; default
!< `csr`).
use mpi
use, intrinsic :: ieee_exceptions, only : ieee_usual, ieee_get_halting_mode, ieee_set_halting_mode
use, intrinsic :: iso_fortran_env, only : int64
use elisha, only : I4P, R_P, ELISHA_OK, ELISHA_BC_DIRICHLET, ELISHA_BC_NEUMANN, ELISHA_BC_DIRICHLET_GHOST, &
                   ELISHA_OPERATOR_CSR, ELISHA_OPERATOR_STENCIL, ELISHA_VCYCLE_ADAM, &
                   elisha_comm_object, elisha_mesh_object, elisha_hierarchical_solver_object
use adam_grid_object, only : grid_object
use adam_field_object, only : field_object
use adam_mpih_global, only : mpih
use adam_poisson_mg_object, only : poisson_mg_object
implicit none
real(R_P),    parameter  :: FACTOR=4._R_P * acos(-1._R_P) !< Source factor (4 pi G, G = 1).
integer(I4P), parameter  :: D=ELISHA_BC_DIRICHLET, NM=ELISHA_BC_NEUMANN, G=ELISHA_BC_DIRICHLET_GHOST !< Kinds.
type(elisha_comm_object) :: c          !< Communication context.
integer(I4P)             :: stat, reps !< Status, repetitions.
integer                  :: ierr, na   !< MPI error status, arguments.
logical                  :: halting(3) !< The FP halting modes of the build.
logical                  :: efirst=.false. !< ELISHA first in every pair.
character(:), allocatable :: msg       !< Message.
character(16)            :: arg        !< Argument.
character(16)            :: cases='abcd' !< The cases to run.
integer(I4P)             :: lop=ELISHA_OPERATOR_CSR !< ELISHA's level operators.

call ieee_get_halting_mode(ieee_usual, halting)
call ieee_set_halting_mode(ieee_usual, .false.)
call MPI_INIT(ierr)
call ieee_set_halting_mode(ieee_usual, halting)
call mpih%initialize
call c%init(MPI_COMM_WORLD, stat, msg)
if (stat /= ELISHA_OK) error stop 'comm init'
reps = 15
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
   print '(A,I0,A,I0,A,A,A,A)', '# elisha_bench_hierarchical_mpi: ', c%nranks, ' rank(s), ', reps, &
         ' repetitions (max over ranks), order ', merge('E, A', 'A, E', efirst), ', ELISHA level operators ', &
         trim(merge('stencil', 'csr    ', lop == ELISHA_OPERATOR_STENCIL))
   print '(A)', '# case             cells  lv     it root setup_A[s] setup_E[s] solve_A[s] solve_E[s] E/A(min) '// &
                ' med_A[s]   med_E[s]  E/A(med) bitwise'
endif
if (index(cases, 'a') > 0) call run_case('(a) AMR periodic ', .true., periodic=.true., isolated=.false.)
if (index(cases, 'b') > 0) call run_case('(b) AMR ghost-D  ', .true., periodic=.false., isolated=.false.)
if (index(cases, 'c') > 0) call run_case('(c) AMR isolated ', .true., periodic=.false., isolated=.true.)
if (index(cases, 'd') > 0) call run_case('(d) D/N root 32^3', .false., periodic=.false., isolated=.false., &
                                          bc=reshape([D, NM, NM, D, D, D], [2, 3]), &
              root_cells=32768_I4P)
call c%destroy(stat)
call MPI_FINALIZE(ierr)

contains
   subroutine run_case(label, amr, periodic, isolated, bc, root_cells)
   !< One case: the level-3 octree of [-1,1]^3 (8^3 blocks of 8^3 cells), the central box refined when `amr`; both
   !< setups, then the alternated cold solves.
   character(*), intent(in)           :: label      !< Case label.
   logical,      intent(in)           :: amr        !< Refine the central box once.
   logical,      intent(in)           :: periodic   !< Periodic axes (all or none).
   logical,      intent(in)           :: isolated   !< Isolated boundaries.
   integer(I4P), intent(in), optional :: bc(2,3)    !< Kinds per face.
   integer(I4P), intent(in), optional :: root_cells !< Most cells of the root.
   integer(I4P), parameter  :: NC=8_I4P, NGC=2_I4P, L0=3_I4P, NB=8_I4P !< Cells per block, ghosts, level, blocks.
   real(R_P),    parameter  :: SIGMA2=0.05_R_P   !< Gaussian width squared (ADAM's).
   type(grid_object)                       :: grid  !< ADAM's grid (shim).
   type(field_object)                      :: field !< ADAM's field (shim).
   type(poisson_mg_object)                 :: adam  !< ADAM's multigrid.
   type(elisha_mesh_object)                :: mesh  !< ELISHA's mesh.
   type(elisha_hierarchical_solver_object) :: eli   !< ELISHA's hierarchical solver.
   integer(I4P), allocatable :: leaves(:,:), c4(:,:)      !< Every leaf, this rank's.
   real(R_P),    allocatable :: bmin(:,:), bmax(:,:)      !< This rank's block corners.
   real(R_P),    allocatable :: buf(:), pa(:,:,:,:), pe(:,:,:,:) !< Source, potentials.
   real(R_P),    allocatable :: ta(:), te(:)              !< Solve times per repetition.
   real(R_P)                 :: dx(3), x(3), sa, se, t0   !< Spacing, cell centre, setup times, mark.
   integer(I4P)              :: nleaf, n, b, gg, i, j, k, m, o, r, slot, nbad(1), lmax, ita, ite !< Counters.
   logical                   :: inbox                     !< A level-3 block inside the central box.

   ! every leaf: level-3 blocks i fastest, the ones inside [-1/2,1/2]^3 (indices 2..5) replaced by their 8 children
   nleaf = 0
   do k=0, NB-1
      do j=0, NB-1
         do i=0, NB-1
            inbox = amr .and. all([i, j, k] >= 2 .and. [i, j, k] <= 5)
            nleaf = nleaf + merge(8, 1, inbox)
         enddo
      enddo
   enddo
   allocate(leaves(4,nleaf))
   gg = 0
   do k=0, NB-1
      do j=0, NB-1
         do i=0, NB-1
            inbox = amr .and. all([i, j, k] >= 2 .and. [i, j, k] <= 5)
            if (inbox) then
               do o=0, 7
                  gg = gg + 1
                  leaves(:,gg) = [2*i + ibits(o, 0, 1), 2*j + ibits(o, 1, 1), 2*k + ibits(o, 2, 1), L0 + 1]
               enddo
            else
               gg = gg + 1
               leaves(:,gg) = [i, j, k, L0]
            endif
         enddo
      enddo
   enddo
   lmax = merge(L0 + 1, L0, amr)
   dx = 2._R_P / NC ! level 0: one block of NC^3 cells on [-1,1]^3
   call grid%set(nijk=[NC, NC, NC], ngc=NGC, nblk0=[1_I4P, 1_I4P, 1_I4P], dx0=dx, emin=[-1._R_P, -1._R_P, -1._R_P], &
                 periodic=spread(periodic, 1, 3), null=[.false., .false., .false.], maxlevel=lmax)
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
   ! the setups, once each
   call MPI_BARRIER(MPI_COMM_WORLD, ierr) ; t0 = MPI_WTIME()
   if (present(bc)) then
      call adam%initialize(grid=grid, field=field, bc=bc, root_cells=root_cells)
   else
      call adam%initialize(grid=grid, field=field, isolated=isolated)
   endif
   sa = tmax(MPI_WTIME() - t0)
   call MPI_BARRIER(MPI_COMM_WORLD, ierr) ; t0 = MPI_WTIME()
   if (present(bc)) then
      call eli%setup(mesh, c, stat, msg, bc=bc, root_cells=root_cells, level_operator=lop, vcycle=ELISHA_VCYCLE_ADAM)
   else
      call eli%setup(mesh, c, stat, msg, isolated=isolated, level_operator=lop, vcycle=ELISHA_VCYCLE_ADAM)
   endif
   se = tmax(MPI_WTIME() - t0)
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
         call MPI_BARRIER(MPI_COMM_WORLD, ierr) ; t0 = MPI_WTIME()
         if ((slot == 1) .neqv. efirst) then
            call adam%solve_packed(field=field, buf=buf, factor=FACTOR, phi=pa)
            ta(r) = tmax(MPI_WTIME() - t0)
         else
            call eli%solve_packed(mesh, buf, FACTOR, pe, stat)
            te(r) = tmax(MPI_WTIME() - t0)
         endif
      enddo
   enddo
   if (c%rank == 0) print '(A,I8,1X,I2,A,I2,I4,1X,A4,2F11.4,2F11.5,F9.4,2F11.5,F9.4,2X,A)', label, mesh%nleaf * NC**3, &
      eli%lbot, '..', eli%lmax, ite, merge('red ', 'slab', eli%fft%redundant), sa, se, minval(ta), minval(te), &
      minval(te) / minval(ta), median(ta), median(te), median(te) / median(ta), merge('yes', 'NO ', nbad(1) == 0)
   call adam%destroy
   call eli%destroy
   call mesh%destroy
   endsubroutine run_case

   function tmax(t) result(tm)
   !< The maximum of t over the ranks.
   real(R_P), intent(in) :: t  !< This rank's time.
   real(R_P)             :: tm !< Maximum.
   integer               :: ierr_ !< MPI error status.

   tm = t
   call MPI_ALLREDUCE(MPI_IN_PLACE, tm, 1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, ierr_)
   endfunction tmax

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
endprogram elisha_bench_hierarchical_mpi
