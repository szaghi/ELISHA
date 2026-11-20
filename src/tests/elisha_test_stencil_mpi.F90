!< ELISHA test: the stencil level operators (`ELISHA_OPERATOR_STENCIL`) agree with the CSR ones to round-off.
program elisha_test_stencil_mpi
!< ELISHA test: the stencil level operators (`ELISHA_OPERATOR_STENCIL`) agree with the CSR ones to round-off.
!<
!< M7, audit C2. Every case sets up two hierarchical solvers on one mesh, `level_operator` CSR (the default) and
!< STENCIL, and compares every level operator applied to a fixed vector (`apply(al(lv), x, y)`) and the red-black
!< smoother from a fixed correction and right-hand side (`smooth`, both orders), within `TOL_OP` relative (max-norm);
!< a full solve, the potential with every ghost layer and the energy within `TOL_PHI`, the iterations within one, the
!< residuals at their stop tolerance; `symmetry_defect` of a fixed pair below `TOL_SYM` for both. It prints whether all
!< of it is bitwise (the stencil takes its rows' CSR order: bitwise with gfortran 16.2 and ifx 2026.2; nvfortran 26.9
!< -fast differs in the last bit of about 1e-16 of the entries) and the share of the level rows the stencil takes. Cases (8^3-cell blocks, 2 ghost layers, the leaves dealt round-robin): uniform periodic with two
!< levels inside the blocks (root 64), anisotropic uniform, 2:1 AMR periodic, AMR with Dirichlet/Neumann faces and
!< boundary values, isolated AMR (the James step), 2-D AMR (null z, root 16). Also the refusal of an unknown
!< `level_operator` (`ELISHA_ERR_ARGUMENT`). Serial build and MPI on any number of ranks.
#ifndef ELISHA_NO_MPI
use mpi
#endif
use, intrinsic :: iso_fortran_env, only : int64
use, intrinsic :: ieee_exceptions, only : ieee_usual, ieee_get_halting_mode, ieee_set_halting_mode
use elisha, only : I4P, R_P, ELISHA_OK, ELISHA_ERR_ARGUMENT, ELISHA_BC_DIRICHLET, ELISHA_BC_NEUMANN, &
                   ELISHA_OPERATOR_CSR, ELISHA_OPERATOR_STENCIL, elisha_comm_object, elisha_mesh_object, &
                   elisha_hierarchical_solver_object
implicit none
real(R_P),    parameter  :: FACTOR=2.5_R_P      !< Source factor.
real(R_P),    parameter  :: TOL=1.e-11_R_P      !< FCG tolerance.
integer(I4P), parameter  :: MAX_IT=100_I4P      !< FCG iterations allowed.
integer(I4P), parameter  :: NG=2_I4P            !< Ghost layers.
real(R_P),    parameter  :: TOL_OP=1.e-14_R_P   !< Level operator and smoother: max |stencil - CSR| / max |CSR|.
real(R_P),    parameter  :: TOL_PHI=1.e-9_R_P   !< Full solve: potential and energy, relative (100 times TOL).
real(R_P),    parameter  :: TOL_SYM=1.e-12_R_P  !< Symmetry defect.
integer(I4P), parameter  :: D=ELISHA_BC_DIRICHLET, NM=ELISHA_BC_NEUMANN !< Kinds.
logical,      parameter  :: T_=.true., F_=.false. !< Shorthands.
real(R_P),    parameter  :: H16(3)=1._R_P / 16._R_P, H24(3)=1._R_P / 24._R_P !< Spacings of unit boxes.
type(elisha_comm_object) :: c                   !< Communication context.
integer(I4P)             :: failures, nfail(1), stat !< Failed checks; status.
integer                  :: wrank, wsize, world, ierr !< MPI.
logical                  :: halting(3)          !< The FP halting modes of the build.
character(:), allocatable :: msg                !< Message.

failures = 0
#ifndef ELISHA_NO_MPI
call ieee_get_halting_mode(ieee_usual, halting)
call ieee_set_halting_mode(ieee_usual, .false.)
call MPI_INIT(ierr)
call ieee_set_halting_mode(ieee_usual, halting)
world = MPI_COMM_WORLD
#else
halting = .false. ; ierr = 0 ; world = 0
#endif
call c%init(world, stat, msg) ; if (stat /= ELISHA_OK) call fail('comm init: '//msg)
wrank = c%rank ; wsize = c%nranks
call run_case('uniform, root 64        ', [2,2,2], [8,8,8], H16, F_, [0,0,0], F_, T_, F_, 64)
call run_case('uniform aniso           ', [2,3,1], [8,8,8], [0.1_R_P, 0.07_R_P, 0.13_R_P], F_, [0,0,0], F_, T_, F_, 0)
call run_case('AMR corner periodic     ', [2,2,2], [8,8,8], H16, T_, [0,0,0], F_, T_, F_, 0)
call run_case('AMR D/N,N/D,D/D + g     ', [2,2,2], [8,8,8], H16, T_, [0,1,0], F_, F_, F_, 0, &
              bc=reshape([D,NM, NM,D, D,D], [2,3]))
call run_case('isolated AMR            ', [3,3,3], [8,8,8], H24, T_, [1,1,1], F_, F_, T_, 0)
call run_case('2-D AMR periodic, r16   ', [2,2,1], [8,8,1], H16, T_, [0,1,0], T_, T_, F_, 16)
call check_refusal
nfail = failures
call c%max_i4(nfail, stat)
call c%destroy(stat)
#ifndef ELISHA_NO_MPI
call MPI_FINALIZE(ierr)
#endif
if (nfail(1) > 0) error stop 1
if (wrank == 0) print '(A,I0,A)', 'elisha_test_stencil_mpi: ok (', wsize, ' rank(s): stencil = CSR to round-off on every case)'

contains
   subroutine make_mesh(mesh, nblk0, nijk, dx0, amr, refine, null2d, periodic)
   !< The mesh: `nblk0` level-0 blocks, block `refine` refined to level 1 when `amr`; the leaves dealt round-robin.
   type(elisha_mesh_object), intent(inout) :: mesh      !< Mesh.
   integer,                  intent(in)    :: nblk0(3)  !< Blocks per axis at level 0.
   integer,                  intent(in)    :: nijk(3)   !< Cells per block.
   real(R_P),                intent(in)    :: dx0(3)    !< Cell spacing at level 0.
   logical,                  intent(in)    :: amr       !< Refine block `refine`.
   integer,                  intent(in)    :: refine(3) !< The refined block (0-based).
   logical,                  intent(in)    :: null2d    !< Null z axis.
   logical,                  intent(in)    :: periodic  !< Every active axis periodic (else none).
   integer(I4P), allocatable :: leaves(:,:), coords(:,:) !< Every leaf, this rank's.
   integer(I4P)              :: nleaf, n, i, j, k, o, b !< Counters.

   nleaf = product(nblk0) + merge(merge(3, 7, null2d), 0, amr)
   allocate(leaves(4,nleaf))
   n = 0
   do k=0, nblk0(3)-1 ; do j=0, nblk0(2)-1 ; do i=0, nblk0(1)-1
      if (amr .and. all([i, j, k] == refine)) then
         do o=0, 7
            if (null2d .and. ibits(o, 2, 1) == 1) cycle
            n = n + 1
            leaves(:,n) = [2*i + ibits(o, 0, 1), 2*j + ibits(o, 1, 1), merge(k, 2*k + ibits(o, 2, 1), null2d), 1]
         enddo
      else
         n = n + 1 ; leaves(:,n) = [i, j, k, 0]
      endif
   enddo ; enddo ; enddo
   allocate(coords(4,count([(mod(n, wsize) == wrank, n=0, nleaf-1)])))
   b = 0
   do n=0, nleaf-1
      if (mod(n, wsize) /= wrank) cycle
      b = b + 1 ; coords(:,b) = leaves(:,n+1)
   enddo
   call mesh%init(int(nijk, I4P), NG, int(nblk0, I4P), dx0, [0._R_P, 0._R_P, 0._R_P], &
                  [periodic, periodic, periodic .and. .not.null2d], [F_, F_, null2d], coords, stat, msg)
   if (stat /= ELISHA_OK) call fail('mesh init: '//msg)
   endsubroutine make_mesh

   subroutine run_case(label, nblk0, nijk, dx0, amr, refine, null2d, periodic, isolated, root_cells, bc)
   !< One case: the two solvers, every comparison.
   character(*), intent(in)           :: label      !< Case label.
   integer,      intent(in)           :: nblk0(3)   !< Blocks per axis at level 0.
   integer,      intent(in)           :: nijk(3)    !< Cells per block.
   real(R_P),    intent(in)           :: dx0(3)     !< Spacing at level 0.
   logical,      intent(in)           :: amr        !< AMR.
   integer,      intent(in)           :: refine(3)  !< The refined block.
   logical,      intent(in)           :: null2d     !< Null z axis.
   logical,      intent(in)           :: periodic   !< Periodic.
   logical,      intent(in)           :: isolated   !< Isolated.
   integer,      intent(in)           :: root_cells !< Root cap (0: default).
   integer(I4P), intent(in), optional :: bc(2,3)    !< Kinds.
   type(elisha_mesh_object)                :: mesh   !< Mesh.
   type(elisha_hierarchical_solver_object) :: sc, ss !< CSR and stencil solvers.
   real(R_P), allocatable :: x(:), yc(:), ys(:), ec(:), es(:), rhs(:), buf(:), pc(:,:,:,:), ps(:,:,:,:), gv(:), z(:)
   real(R_P)              :: ecs, ess, symc, syms, rel(1), dmax(1), rphi(1) !< Energies, defects, differences.
   integer(I4P)           :: lv, i, n, nbad(1), order, rows(2) !< Level, counters, differing bits, rows.

   call make_mesh(mesh, nblk0, nijk, dx0, amr, refine, null2d, periodic)
   if (root_cells > 0) then
      call sc%setup(mesh, c, stat, msg, tolerance=TOL, max_iterations=MAX_IT, isolated=isolated, bc=bc, &
                    root_cells=int(root_cells, I4P))
      if (stat == ELISHA_OK) call ss%setup(mesh, c, stat, msg, tolerance=TOL, max_iterations=MAX_IT, isolated=isolated, &
                                           bc=bc, root_cells=int(root_cells, I4P), level_operator=ELISHA_OPERATOR_STENCIL)
   else
      call sc%setup(mesh, c, stat, msg, tolerance=TOL, max_iterations=MAX_IT, isolated=isolated, bc=bc)
      if (stat == ELISHA_OK) call ss%setup(mesh, c, stat, msg, tolerance=TOL, max_iterations=MAX_IT, isolated=isolated, &
                                           bc=bc, level_operator=ELISHA_OPERATOR_STENCIL)
   endif
   if (stat /= ELISHA_OK) then
      call fail(label//': setup: '//msg) ; return
   endif
   nbad = 0 ; rel = 0._R_P ; rows = 0
   ! every level operator on a fixed vector, and the smoother
   do lv=sc%lbot+1, sc%lmax
      n = sc%al(lv)%nloc
      if (.not.ss%al(lv)%stencil) call fail(label//': a level operator without its stencil')
      rows(1) = rows(1) + sc%al(lv)%nrow
      rows(2) = rows(2) + ss%al(lv)%snblk * max(0, ss%al(lv)%shi(1) - ss%al(lv)%slo(1) + 1) * &
                max(0, ss%al(lv)%shi(2) - ss%al(lv)%slo(2) + 1) * max(0, ss%al(lv)%shi(3) - ss%al(lv)%slo(3) + 1)
      allocate(x(max(n,1)), yc(max(n,1)), ys(max(n,1)), ec(max(n,1)), es(max(n,1)), rhs(max(n,1)))
      do i=1, size(x)
         x(i) = sin(1.3_R_P * i + 0.7_R_P * wrank + lv) ; rhs(i) = cos(2.1_R_P * i - 0.4_R_P * wrank)
      enddo
      yc = 0._R_P ; ys = 0._R_P
      call sc%apply(sc%al(lv), x, yc, stat) ; if (stat /= ELISHA_OK) call fail(label//': apply CSR')
      call ss%apply(ss%al(lv), x, ys, stat) ; if (stat /= ELISHA_OK) call fail(label//': apply stencil')
      nbad = nbad + differ(yc, ys) ; rel = max(rel, reldiff(yc, ys))
      do order=0, 1
         ec = x ; es = x
         call sc%smooth(lv, ec, rhs, order, stat) ; if (stat /= ELISHA_OK) call fail(label//': smooth CSR')
         call ss%smooth(lv, es, rhs, order, stat) ; if (stat /= ELISHA_OK) call fail(label//': smooth stencil')
         nbad = nbad + differ(ec, es) ; rel = max(rel, reldiff(ec, es))
      enddo
      deallocate(x, yc, ys, ec, es, rhs)
   enddo
   ! a full solve
   allocate(buf(max(mesh%nb*product(nijk),1)))
   do i=1, size(buf)
      buf(i) = 0.7_R_P + cos(0.37_R_P * i + 0.2_R_P * wrank) + 0.3_R_P * sin(0.011_R_P * i * i)
   enddo
   allocate(pc(1-NG:nijk(1)+NG,1-NG:nijk(2)+NG,1-NG:nijk(3)+NG,1:max(mesh%nb,1)))
   allocate(ps, mold=pc) ; pc = 0._R_P ; ps = 0._R_P
   allocate(gv(max(sc%nbnd,1)))
   gv = 0._R_P
   do i=1, sc%nbnd
      gv(i) = 0.3_R_P + sin(3._R_P * sc%bnd_x(1,i)) + 0.2_R_P * sc%bnd_x(2,i)
   enddo
   if (present(bc)) then
      call sc%solve_packed(mesh, buf, FACTOR, pc, stat, msg, energy=ecs, ghost=gv)
      if (stat == ELISHA_OK) call ss%solve_packed(mesh, buf, FACTOR, ps, stat, msg, energy=ess, ghost=gv)
   else
      call sc%solve_packed(mesh, buf, FACTOR, pc, stat, msg, energy=ecs)
      if (stat == ELISHA_OK) call ss%solve_packed(mesh, buf, FACTOR, ps, stat, msg, energy=ess)
   endif
   if (stat /= ELISHA_OK) then
      call fail(label//': solve: '//msg) ; return
   endif
   nbad = nbad + differ(reshape(pc, [size(pc)]), reshape(ps, [size(ps)])) + differ([ecs, sc%residual], [ess, ss%residual])
   if (sc%iterations /= ss%iterations .or. sc%iterations0 /= ss%iterations0) nbad = nbad + 1
   rphi = max(reldiff(reshape(pc, [size(pc)]), reshape(ps, [size(ps)])), abs(ess - ecs) / max(abs(ecs), tiny(1._R_P)))
   if (abs(ss%iterations - sc%iterations) > 1 .or. abs(ss%iterations0 - sc%iterations0) > 1) &
      call fail(label//': the iterations differ by more than one')
   if (.not.(ss%residual <= ss%tolerance_used .and. sc%residual <= sc%tolerance_used)) &
      call fail(label//': a residual above its stop tolerance')
   ! the operator's symmetry
   allocate(x(max(mesh%nb*product(nijk),1)), z(max(mesh%nb*product(nijk),1)))
   do i=1, size(x)
      x(i) = sin(1.3_R_P * i + 0.7_R_P * wrank) ; z(i) = cos(2.1_R_P * i - 0.4_R_P * wrank)
   enddo
   symc = sc%symmetry_defect(x, z, stat) ; syms = ss%symmetry_defect(x, z, stat)
   nbad = nbad + differ([symc], [syms])
   if (.not.(syms <= TOL_SYM .and. symc <= TOL_SYM)) call fail(label//': symmetry defect above TOL_SYM')
   dmax = rel ; call c%max_real8(dmax, stat)
   call c%max_real8(rphi, stat)
   call c%max_i4(nbad, stat)
   call c%max_i4(rows, stat)
   if (wrank == 0) print '(A,I0,A,I0,A,I0,A,I0,A,I0,A,ES9.2,A,ES9.2,A,ES9.2,A,A)', 'elisha_test_stencil_mpi: '//label// &
      ' np ', wsize, ', levels ', sc%lbot+1, '..', sc%lmax, ', iterations ', sc%iterations, '/', ss%iterations, &
      ', operators ', dmax(1), ', phi ', rphi(1), ', symmetry ', syms, ', bitwise ', merge('yes', 'no ', nbad(1) == 0)
   if (wrank == 0) print '(A,I0,A,I0,A)', '                         stencil rows (rank max) ', rows(2), ' of ', rows(1), &
      ' level rows'
   if (.not.(dmax(1) <= TOL_OP)) call fail(label//': a level operator or smoother differs above TOL_OP')
   if (.not.(rphi(1) <= TOL_PHI)) call fail(label//': the potential or the energy differs above TOL_PHI')
   call sc%destroy ; call ss%destroy ; call mesh%destroy
   endsubroutine run_case

   subroutine check_refusal
   !< An unknown level_operator.
   type(elisha_mesh_object)                :: mesh   !< Mesh.
   type(elisha_hierarchical_solver_object) :: s      !< Solver.

   call make_mesh(mesh, [2,2,2], [8,8,8], H16, F_, [0,0,0], F_, T_)
   call s%setup(mesh, c, stat, msg, level_operator=7_I4P)
   if (stat /= ELISHA_ERR_ARGUMENT) then
      call fail('an unknown level_operator was not refused')
   elseif (wrank == 0) then
      print '(A)', 'elisha_test_stencil_mpi: as expected: an unknown level_operator ('//msg//')'
   endif
   call s%setup(mesh, c, stat, msg, level_operator=ELISHA_OPERATOR_CSR)
   if (stat /= ELISHA_OK) call fail('level_operator CSR refused')
   call s%destroy ; call mesh%destroy
   endsubroutine check_refusal

   pure function differ(a, b) result(nbad)
   !< The entries of two real arrays whose bit patterns differ.
   real(R_P), intent(in) :: a(:), b(:) !< Arrays.
   integer(I4P)          :: nbad       !< Differing entries.

   nbad = count(transfer(a, [0_int64], size(a)) /= transfer(b, [0_int64], size(b)))
   endfunction differ

   pure function reldiff(a, b) result(r)
   !< max |a - b| / max |a|.
   real(R_P), intent(in) :: a(:), b(:) !< Arrays.
   real(R_P)             :: r          !< Relative difference.

   r = maxval(abs(a - b)) / max(maxval(abs(a)), tiny(1._R_P))
   endfunction reldiff

   subroutine fail(what)
   !< Report a failed check.
   character(*), intent(in) :: what !< What failed.

   failures = failures + 1
   print '(A,I0,A)', '  FAIL rank ', wrank, ': '//what
   endsubroutine fail
endprogram elisha_test_stencil_mpi
