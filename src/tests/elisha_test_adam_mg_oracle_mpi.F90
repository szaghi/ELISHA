!< ELISHA test: ADAM's multigrid Poisson solver runs inside ELISHA's tests as an oracle, and solves what it claims.
program elisha_test_adam_mg_oracle_mpi
!< ELISHA test: ADAM's multigrid Poisson solver runs inside ELISHA's tests as an oracle, and solves what it claims.
!<
!< The solver is `adam_poisson_mg_object`, ADAM 669d0bdd verbatim (its root `adam_poisson_fft_object` too), over the
!< test shims `adam_grid_object`, `adam_field_object`, `adam_mpih_global`. Periodic 3-D grids, the leaves dealt
!< round-robin to the ranks, the source a constant plus a few smooth periodic modes sampled at the cell centres:
!< - (a) uniform, 2x2x2 blocks of 8^3 cells, the root capped at 64 cells (two levels inside the blocks: the V-cycle
!<   smooths, it is no exact solve);
!< - (b) 2:1 AMR: a 2x2x2 level-0 domain with block (0,0,0) refined to level 1 (its children meet level-0 leaves
!<   across the periodic wrap); a 3x3x3 level-0 domain with the centre block refined (it touches no domain face), the
!<   root capped at 216 cells (levels inside the blocks below the tree);
!< - (c) uniform, the default root (the whole level-0 domain, one FFT solve: one V-cycle is the exact solve, ADAM's
!<   flail.md), cubic and anisotropic cells.
!< Checked by meaning, on every case: the FCG residual ADAM reports at most its stop tolerance (`tolerance_used`, the
!< requested one or the source's round-off floor); at least one iteration; the residual recomputed through ADAM's own
!< `apply_a`, `max |(A phi) / V - factor (f - mean f)| / max |factor (f - mean f)|` (the mean volume-weighted), at most
!< twice the stop tolerance; ADAM's `symmetry_defect` of two fixed vectors at most 1e-12. On (c): one iteration, and
!< the potential against ADAM's FFT solver's, `max |phi_mg - phi_fft| / max |phi_fft|` at most 1e-12 (printed, with
!< the cells where the two agree bitwise). ADAM's solvers need MPI: built with `-DELISHA_NO_MPI` (the serial modes, fpm)
!< the program prints a skip line and succeeds. The FP traps of a debug build are off around `MPI_Init` only (hwloc
!< raises IEEE flags inside it).
#ifndef ELISHA_NO_MPI
use :: mpi
use, intrinsic :: ieee_exceptions, only : ieee_usual, ieee_get_halting_mode, ieee_set_halting_mode
use :: penf, only : I4P, I8P, R8P
use :: adam_grid_object, only : grid_object
use :: adam_field_object, only : field_object
use :: adam_mpih_global, only : mpih
use :: adam_poisson_fft_object, only : poisson_fft_object
use :: adam_poisson_mg_object, only : poisson_mg_object
implicit none
real(R8P),    parameter :: TOL_MG=1.e-11_R8P     !< FCG tolerance requested.
integer(I4P), parameter :: MAX_IT=100_I4P        !< FCG iterations allowed.
real(R8P),    parameter :: TOL_SYMMETRY=1.e-12_R8P !< Bound on the operator's symmetry defect.
real(R8P),    parameter :: TOL_FFT=1.e-12_R8P    !< Bound on max |phi_mg - phi_fft| / max |phi_fft| (uniform).
real(R8P),    parameter :: FACTOR=2.5_R8P        !< Source factor.
real(R8P),    parameter :: PI=acos(-1._R8P)      !< Pi.
real(R8P),    parameter :: C0=0.7_R8P            !< The constant of the source.
integer(I4P), parameter :: NGHOST=2_I4P          !< Ghost layers.
integer(I4P)            :: failures, nfail       !< Failed checks, on this rank and over the ranks.
integer                 :: ierr                  !< MPI error status.
logical                 :: halting(3)            !< The FP halting modes of the build (debug: on).

failures = 0
! MPI_Init runs C code (hwloc's XML backend) that raises IEEE flags: a debug build's FP traps (-ffpe-trap) would
! kill it there, so they are off around the call only (Open MPI 5.0.11, hwloc 2.7.1)
call ieee_get_halting_mode(ieee_usual, halting)
call ieee_set_halting_mode(ieee_usual, .false.)
call MPI_INIT(ierr)
call ieee_set_halting_mode(ieee_usual, halting)
call mpih%initialize
call run_case(label='(a) uniform, root 64     ', nblk0=[2_I4P, 2_I4P, 2_I4P], dx0=[1._R8P, 1._R8P, 1._R8P] / 16._R8P, &
              amr=.false., refine=[0_I4P, 0_I4P, 0_I4P], root_cells=64_I4P, versus_fft=.false.)
call run_case(label='(b) AMR 2^3, corner      ', nblk0=[2_I4P, 2_I4P, 2_I4P], dx0=[1._R8P, 1._R8P, 1._R8P] / 16._R8P, &
              amr=.true., refine=[0_I4P, 0_I4P, 0_I4P], root_cells=0_I4P, versus_fft=.false.)
call run_case(label='(b) AMR 3^3, centre, r216', nblk0=[3_I4P, 3_I4P, 3_I4P], dx0=[1._R8P, 1._R8P, 1._R8P] / 24._R8P, &
              amr=.true., refine=[1_I4P, 1_I4P, 1_I4P], root_cells=216_I4P, versus_fft=.false.)
call run_case(label='(c) uniform vs FFT, cubic', nblk0=[2_I4P, 2_I4P, 2_I4P], dx0=[1._R8P, 1._R8P, 1._R8P] / 16._R8P, &
              amr=.false., refine=[0_I4P, 0_I4P, 0_I4P], root_cells=0_I4P, versus_fft=.true.)
call run_case(label='(c) uniform vs FFT, aniso', nblk0=[2_I4P, 3_I4P, 1_I4P], dx0=[0.1_R8P, 0.07_R8P, 0.13_R8P], &
              amr=.false., refine=[0_I4P, 0_I4P, 0_I4P], root_cells=0_I4P, versus_fft=.true.)
nfail = failures
call MPI_ALLREDUCE(MPI_IN_PLACE, nfail, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)
if (mpih%myrank == 0 .and. nfail == 0) print '(A,I0,A)', 'elisha_test_adam_mg_oracle_mpi: ok (', mpih%procs_number, &
                                                        ' rank(s))'
call MPI_FINALIZE(ierr)
if (nfail > 0) error stop 1

contains
   subroutine run_case(label, nblk0, dx0, amr, refine, root_cells, versus_fft)
   !< One grid: 8^3 cells per block, `nblk0` level-0 blocks, block `refine` refined to level 1 when `amr`; the leaves
   !< dealt round-robin; ADAM's multigrid solve and its checks by meaning; with `versus_fft`, ADAM's FFT solve compared.
   character(*), intent(in)  :: label      !< Case label.
   integer(I4P), intent(in)  :: nblk0(3)   !< Blocks per axis at level 0.
   real(R8P),    intent(in)  :: dx0(3)     !< Cell spacing at level 0.
   logical,      intent(in)  :: amr        !< Refine block `refine` to level 1.
   integer(I4P), intent(in)  :: refine(3)  !< The refined level-0 block (0-based).
   integer(I4P), intent(in)  :: root_cells !< Most cells of the multigrid's root (0: ADAM's default).
   logical,      intent(in)  :: versus_fft !< Compare the potential with ADAM's FFT solver's.
   integer(I4P), parameter   :: NIJK(3)=[8_I4P, 8_I4P, 8_I4P] !< Cells per block per axis.
   type(grid_object)         :: grid       !< ADAM's grid (shim).
   type(field_object)        :: field      !< ADAM's field (shim).
   type(poisson_mg_object)   :: mg         !< ADAM's multigrid solver.
   type(poisson_fft_object)  :: fft        !< ADAM's FFT solver.
   integer(I4P), allocatable :: leaves(:,:), coords(:,:) !< Every leaf, this rank's leaves [4,*].
   real(R8P),    allocatable :: bmin(:,:), bmax(:,:)     !< This rank's block corners [3,nb].
   real(R8P),    allocatable :: q(:,:,:,:,:), phi(:,:,:,:), phi_fft(:,:,:,:) !< Source, potentials.
   real(R8P),    allocatable :: u(:), au(:), x(:), y(:)  !< Packed potential, A u, symmetry vectors.
   integer(I4P)              :: nleaf, nb, g, b, i, j, k, c, o, lmax, iterations !< Counters.
   integer(I4P)              :: same                     !< Cells where the MG and FFT potentials agree bitwise.
   real(R8P)                 :: dx(3), xc(3), vol, s(2), mean, res(2), sym, cmp(2) !< Measures.
   logical                   :: is_ref   !< The level-0 block is the refined one.
   integer                   :: ierr_    !< MPI error status.

   ! every leaf, level-0 blocks i fastest, the refined one replaced by its 8 children; filled in place
   nleaf = product(nblk0) + merge(7_I4P, 0_I4P, amr)
   allocate(leaves(4,nleaf))
   g = 0
   do k=0, nblk0(3)-1
      do j=0, nblk0(2)-1
         do i=0, nblk0(1)-1
            is_ref = amr .and. all([i, j, k] == refine)
            if (is_ref) then
               do o=0, 7
                  g = g + 1
                  leaves(:,g) = [2*i + ibits(o, 0, 1), 2*j + ibits(o, 1, 1), 2*k + ibits(o, 2, 1), 1_I4P]
               enddo
            else
               g = g + 1
               leaves(:,g) = [i, j, k, 0_I4P]
            endif
         enddo
      enddo
   enddo
   lmax = merge(1_I4P, 0_I4P, amr)
   call grid%set(nijk=NIJK, ngc=NGHOST, nblk0=nblk0, dx0=dx0, emin=[0._R8P, 0._R8P, 0._R8P], &
                 periodic=[.true., .true., .true.], null=[.false., .false., .false.], maxlevel=lmax)
   ! deal the leaves round-robin: leaf g (0-based) to rank mod(g, ranks); fill in place
   nb = 0
   do g=0, nleaf-1
      if (mod(g, mpih%procs_number) == mpih%myrank) nb = nb + 1
   enddo
   allocate(coords(4,nb), bmin(3,nb), bmax(3,nb))
   b = 0
   do g=0, nleaf-1
      if (mod(g, mpih%procs_number) /= mpih%myrank) cycle
      b = b + 1
      coords(:,b) = leaves(:,g+1)
      bmin(:,b) = grid%domain_emin + coords(1:3,b) * grid%block_dxyz(:,coords(4,b))
      bmax(:,b) = bmin(:,b) + grid%block_dxyz(:,coords(4,b))
   enddo
   call field%set(coords, emin=bmin, emax=bmax)
   allocate(q(1:1,1-NGHOST:NIJK(1)+NGHOST,1-NGHOST:NIJK(2)+NGHOST,1-NGHOST:NIJK(3)+NGHOST,1:max(nb,1)))
   allocate(phi(1-NGHOST:NIJK(1)+NGHOST,1-NGHOST:NIJK(2)+NGHOST,1-NGHOST:NIJK(3)+NGHOST,1:max(nb,1)))
   q = 0._R8P ; phi = 0._R8P
   ! the source at the cell centres; the volume-weighted mean as C0 plus the mean deviation from it
   s = 0._R8P
   do b=1, nb
      dx = grid%cell_dxyz(:,coords(4,b))
      vol = product(dx)
      do k=1, NIJK(3)
         do j=1, NIJK(2)
            do i=1, NIJK(1)
               xc = bmin(:,b) + ([i, j, k] - 0.5_R8P) * dx
               q(1,i,j,k,b) = source(xc, grid%domain_emax - grid%domain_emin)
               s = s + [vol * (q(1,i,j,k,b) - C0), vol]
            enddo
         enddo
      enddo
   enddo
   call MPI_ALLREDUCE(MPI_IN_PLACE, s, 2, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr_)
   mean = C0 + s(1) / s(2)
   if (root_cells > 0) then
      call mg%initialize(grid=grid, field=field, tolerance=TOL_MG, max_iterations=MAX_IT, root_cells=root_cells)
   else
      call mg%initialize(grid=grid, field=field, tolerance=TOL_MG, max_iterations=MAX_IT)
   endif
   call mg%solve(field=field, q=q, iv=1_I4P, factor=FACTOR, phi=phi)
   iterations = mg%iterations
   ! the residual recomputed through ADAM's operator: res(1) max |(A u) / V - factor (f - mean)|, res(2) the scale
   allocate(u(max(nb*product(NIJK),1)), au(max(nb*product(NIJK),1)), x(max(nb*product(NIJK),1)), &
            y(max(nb*product(NIJK),1)))
   u = 0._R8P ; au = 0._R8P
   c = 0
   do b=1, nb
      do k=1, NIJK(3)
         do j=1, NIJK(2)
            do i=1, NIJK(1)
               c = c + 1
               u(c) = phi(i,j,k,b)
            enddo
         enddo
      enddo
   enddo
   call mg%apply_a(u, au)
   res = 0._R8P
   c = 0
   do b=1, nb
      vol = product(grid%cell_dxyz(:,coords(4,b)))
      do k=1, NIJK(3)
         do j=1, NIJK(2)
            do i=1, NIJK(1)
               c = c + 1
               res(1) = max(res(1), abs(au(c) / vol - FACTOR * (q(1,i,j,k,b) - mean)))
               res(2) = max(res(2), abs(FACTOR * (q(1,i,j,k,b) - mean)))
            enddo
         enddo
      enddo
   enddo
   call MPI_ALLREDUCE(MPI_IN_PLACE, res, 2, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, ierr_)
   do i=1, size(x)
      x(i) = sin(1.3_R8P * i + 0.7_R8P * mpih%myrank) ; y(i) = cos(2.1_R8P * i - 0.4_R8P * mpih%myrank)
   enddo
   sym = mg%symmetry_defect(x, y)
   if (mpih%myrank == 0) print '(A,I0,A,I0,A,I0,A,I0,A,I0,A,I0,A,ES9.2,A,ES9.2,A,ES9.2,A,ES9.2)', &
      'elisha_test_adam_mg_oracle_mpi: '//label//' np ', mpih%procs_number, ' leaves ', nleaf, ' levels ', mg%lbot, &
      '..', mg%lmax, ' (lmin ', mg%lmin, ') iterations ', iterations, ' residual ', mg%residual, ' tol ', &
      mg%tolerance_used, ' recomputed ', res(1) / res(2), ' symmetry ', sym
   if (.not.(mg%residual <= mg%tolerance_used)) call fail(label//': FCG residual above its stop tolerance')
   if (.not.(iterations > 0)) call fail(label//': no FCG iteration')
   if (.not.(res(1) <= 2._R8P * mg%tolerance_used * res(2))) call fail(label//': recomputed residual above 2 tol')
   if (.not.(sym <= TOL_SYMMETRY)) call fail(label//': symmetry defect above 1e-12')
   if (versus_fft) then
      if (iterations /= 1) call fail(label//': the exact V-cycle took more than one FCG iteration')
      allocate(phi_fft, mold=phi)
      phi_fft = 0._R8P
      call fft%initialize(grid=grid, field=field)
      call fft%solve(field=field, q=q, iv=1_I4P, factor=FACTOR, phi=phi_fft)
      cmp = 0._R8P
      same = 0
      do b=1, nb
         do k=1, NIJK(3)
            do j=1, NIJK(2)
               do i=1, NIJK(1)
                  cmp(1) = max(cmp(1), abs(phi(i,j,k,b) - phi_fft(i,j,k,b)))
                  cmp(2) = max(cmp(2), abs(phi_fft(i,j,k,b)))
                  if (transfer(phi(i,j,k,b), 0_I8P) == transfer(phi_fft(i,j,k,b), 0_I8P)) same = same + 1
               enddo
            enddo
         enddo
      enddo
      call MPI_ALLREDUCE(MPI_IN_PLACE, cmp, 2, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, ierr_)
      call MPI_ALLREDUCE(MPI_IN_PLACE, same, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr_)
      if (mpih%myrank == 0) print '(A,ES9.2,A,ES9.2,A,I0,A,I0)', 'elisha_test_adam_mg_oracle_mpi: '//label// &
         ' max|phi_mg - phi_fft| ', cmp(1), ' relative ', cmp(1) / cmp(2), ' bitwise-equal cells ', same, ' of ', &
         nleaf * product(NIJK)
      if (.not.(cmp(1) <= TOL_FFT * cmp(2))) call fail(label//': multigrid potential differs from the FFT one')
      call fft%destroy
   endif
   call mg%destroy
   endsubroutine run_case

   pure function source(xc, length) result(f)
   !< The source at the point xc of a periodic box of sides `length` (lower corner at the origin): a constant plus
   !< smooth periodic modes.
   real(R8P), intent(in) :: xc(3)     !< Point.
   real(R8P), intent(in) :: length(3) !< Box sides.
   real(R8P)             :: f         !< Source.
   real(R8P)             :: t(3)      !< Phases per unit wavenumber.

   t = 2._R8P * PI * xc / length
   f = C0 + cos(t(1)) + 0.5_R8P * cos(2._R8P * t(2) + t(3) + 0.3_R8P) &
       + 0.25_R8P * sin(3._R8P * t(1) + t(2) + 2._R8P * t(3))
   endfunction source

   subroutine fail(msg)
   !< Report a failed check.
   character(*), intent(in) :: msg !< Message.

   failures = failures + 1
   print '(A)', 'FAIL: '//msg
   endsubroutine fail
#else
implicit none
print '(A)', 'elisha_test_adam_mg_oracle_mpi: skipped (ADAM''s multigrid solver needs MPI; built with -DELISHA_NO_MPI)'
#endif
endprogram elisha_test_adam_mg_oracle_mpi
