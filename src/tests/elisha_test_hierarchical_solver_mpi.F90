!< ELISHA test: the hierarchical solver solves what it claims on any number of ranks (and without MPI), and refuses.
program elisha_test_hierarchical_solver_mpi
!< ELISHA test: the hierarchical solver solves what it claims on any number of ranks (and without MPI), and refuses.
!<
!< Smoke test by meaning (the parity with ADAM's multigrid is another test's). 3-D meshes of 8^3-cell blocks, the leaves
!< dealt round-robin to the ranks, the source a constant plus smooth periodic modes at the cell centres, through
!< `setup` and `solve_packed`:
!< - (a) uniform periodic, 2x2x2 blocks, the root capped at 64 cells (two levels inside the blocks);
!< - (b) 2:1 AMR periodic: a 2x2x2 level-0 domain with block (0,0,0) refined to level 1; a 3x3x3 one with the centre
!<   block refined (it touches no domain face), the root capped at 216 cells;
!< - (c) uniform periodic, the default root (one V-cycle is the exact solve), cubic and anisotropic cells;
!< - (d) the 2x2x2 AMR mesh non-periodic, ghost-Dirichlet faces (the default), homogeneous;
!< - (e) uniform 2x2x2 isolated (the James step).
!< Checked on every case: the FCG residual at most its stop tolerance (`tolerance_used`), at least one iteration, the
!< operator's `symmetry_defect` of two fixed vectors at most 1e-12, the V-cycle's (`vcycle_symmetry_defect`, the
!< default `ELISHA_VCYCLE_SYMMETRIC`) at most 1e-12; on the AMR cases (b), (d) ADAM's V-cycle (`ELISHA_VCYCLE_ADAM`)
!< set up on the same mesh, its defect at least 1e-6 (ELISHA notes E1c: the seam coupling on the way up only), its
!< FCG iterations printed beside the default's; on (a)-(d) the residual recomputed through
!< `apply_a`, `max |(A phi) / V - factor (f - mean f)| / max |factor (f - mean f)|` (the mean volume-weighted; no mean
!< with a Dirichlet face), at most twice the stop tolerance; on (c) one iteration. Refusals, each with the same status
!< on every rank: a mesh not 2:1 balanced (`ELISHA_ERR_MESH_BALANCE`), isolated with boundary kinds
!< (`ELISHA_ERR_ISOLATED_BC`), a solve before the setup (`ELISHA_ERR_NOT_INITIALIZED`), a `phi` of the wrong shape
!< (`ELISHA_ERR_SHAPE`), fewer ghost values than boundary faces (`ELISHA_ERR_GHOST_COUNT`), a `vcycle` that is no
!< `ELISHA_VCYCLE_*` (`ELISHA_ERR_ARGUMENT`), the device assembly on this host solver (`ELISHA_ERR_ARGUMENT`); and an FCG stopped at `max_iterations` = 1 above its tolerance, on the
!< periodic AMR corner mesh and on the uniform isolated one with its root capped at 64 cells (`ELISHA_ERR_NOT_CONVERGED`, `converged` false, `phi` the last iterate: every
!< value finite, ghosts included, the interior not zero), then the same solver with enough iterations converged. The FP traps of a debug
!< build are off around `MPI_Init` only (hwloc raises IEEE flags inside it).
#ifndef ELISHA_NO_MPI
use mpi
#endif
use, intrinsic :: ieee_exceptions, only : ieee_usual, ieee_get_halting_mode, ieee_set_halting_mode
use elisha, only : I4P, R_P, ELISHA_OK, ELISHA_ERR_MESH_BALANCE, ELISHA_ERR_ISOLATED_BC, ELISHA_ERR_NOT_INITIALIZED, &
                   ELISHA_ERR_SHAPE, ELISHA_ERR_GHOST_COUNT, ELISHA_ERR_ARGUMENT, ELISHA_BC_DIRICHLET_GHOST,          &
                   ELISHA_ERR_NOT_CONVERGED, ELISHA_ASSEMBLY_DEVICE,                                                &
                   ELISHA_VCYCLE_ADAM, elisha_comm_object, elisha_mesh_object, elisha_hierarchical_solver_object
implicit none
real(R_P),    parameter  :: TOL_MG=1.e-11_R_P       !< FCG tolerance requested.
integer(I4P), parameter  :: MAX_IT=100_I4P          !< FCG iterations allowed.
real(R_P),    parameter  :: TOL_SYMMETRY=1.e-12_R_P !< Bound on the operator's and the V-cycle's symmetry defect.
real(R_P),    parameter  :: MIN_ADAM_DEFECT=1.e-6_R_P !< ADAM's V-cycle on AMR: its symmetry defect at least this.
real(R_P),    parameter  :: FACTOR=2.5_R_P          !< Source factor.
real(R_P),    parameter  :: PI=acos(-1._R_P)        !< Pi.
real(R_P),    parameter  :: C0=0.7_R_P              !< The constant of the source.
integer(I4P), parameter  :: NGHOST=2_I4P            !< Ghost layers.
integer(I4P), parameter  :: NIJK(3)=[8_I4P, 8_I4P, 8_I4P] !< Cells per block per axis.
type(elisha_comm_object) :: c                       !< Communication context.
integer(I4P)             :: failures, nfail(1), stat !< Failed checks, over the ranks; status.
integer                  :: wrank, wsize, world, ierr !< MPI.
logical                  :: halting(3)              !< The FP halting modes of the build (debug: on).
character(:), allocatable :: msg                    !< Message.

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
call run_case(label='(a) uniform, root 64     ', nblk0=[2_I4P, 2_I4P, 2_I4P], dx0=[1._R_P, 1._R_P, 1._R_P] / 16._R_P, &
              amr=.false., refine=[0_I4P, 0_I4P, 0_I4P], root_cells=64_I4P, periodic=.true., isolated=.false.)
call run_case(label='(b) AMR 2^3, corner      ', nblk0=[2_I4P, 2_I4P, 2_I4P], dx0=[1._R_P, 1._R_P, 1._R_P] / 16._R_P, &
              amr=.true., refine=[0_I4P, 0_I4P, 0_I4P], root_cells=0_I4P, periodic=.true., isolated=.false.)
call run_case(label='(b) AMR 3^3, centre, r216', nblk0=[3_I4P, 3_I4P, 3_I4P], dx0=[1._R_P, 1._R_P, 1._R_P] / 24._R_P, &
              amr=.true., refine=[1_I4P, 1_I4P, 1_I4P], root_cells=216_I4P, periodic=.true., isolated=.false.)
call run_case(label='(c) uniform, exact root  ', nblk0=[2_I4P, 2_I4P, 2_I4P], dx0=[1._R_P, 1._R_P, 1._R_P] / 16._R_P, &
              amr=.false., refine=[0_I4P, 0_I4P, 0_I4P], root_cells=0_I4P, periodic=.true., isolated=.false.)
call run_case(label='(c) uniform, aniso cells ', nblk0=[2_I4P, 3_I4P, 1_I4P], dx0=[0.1_R_P, 0.07_R_P, 0.13_R_P], &
              amr=.false., refine=[0_I4P, 0_I4P, 0_I4P], root_cells=0_I4P, periodic=.true., isolated=.false.)
call run_case(label='(d) AMR 2^3, ghost-Dir.  ', nblk0=[2_I4P, 2_I4P, 2_I4P], dx0=[1._R_P, 1._R_P, 1._R_P] / 16._R_P, &
              amr=.true., refine=[0_I4P, 0_I4P, 0_I4P], root_cells=0_I4P, periodic=.false., isolated=.false.)
call run_case(label='(e) uniform 2^3, isolated', nblk0=[2_I4P, 2_I4P, 2_I4P], dx0=[1._R_P, 1._R_P, 1._R_P] / 16._R_P, &
              amr=.false., refine=[0_I4P, 0_I4P, 0_I4P], root_cells=0_I4P, periodic=.false., isolated=.true.)
call check_refusals
call check_not_converged(periodic=.true.)
call check_not_converged(periodic=.false.)
nfail = failures
call c%max_i4(nfail, stat)
call c%destroy(stat)
#ifndef ELISHA_NO_MPI
call MPI_FINALIZE(ierr)
#endif
if (nfail(1) > 0) error stop 1
if (wrank == 0) print '(A,I0,A)', 'elisha_test_hierarchical_solver_mpi: ok (', wsize, ' rank(s))'

contains
   subroutine make_mesh(mesh, nblk0, dx0, amr, refine, periodic, extra, stat, msg)
   !< The mesh: `nblk0` level-0 blocks of NIJK cells, block `refine` refined to level 1 when `amr` (and, with `extra`,
   !< its child (1,1,1) refined to level 2: not 2:1 against the level-0 neighbours); the leaves dealt round-robin.
   type(elisha_mesh_object),  intent(inout) :: mesh      !< Mesh.
   integer(I4P),              intent(in)    :: nblk0(3)  !< Blocks per axis at level 0.
   real(R_P),                 intent(in)    :: dx0(3)    !< Cell spacing at level 0.
   logical,                   intent(in)    :: amr       !< Refine block `refine` to level 1.
   integer(I4P),              intent(in)    :: refine(3) !< The refined level-0 block (0-based).
   logical,                   intent(in)    :: periodic  !< Periodic axes (all or none).
   logical,                   intent(in)    :: extra     !< Refine further to break the 2:1 balance.
   integer(I4P),              intent(out)   :: stat      !< Status of the mesh init.
   character(:), allocatable, intent(out)   :: msg       !< Message.
   integer(I4P), allocatable :: leaves(:,:), coords(:,:) !< Every leaf, this rank's leaves [4,*].
   integer(I4P)              :: nleaf, g, i, j, k, o, o2, nb, b !< Counters.

   nleaf = product(nblk0) + merge(7_I4P, 0_I4P, amr) + merge(7_I4P, 0_I4P, extra)
   allocate(leaves(4,nleaf))
   g = 0
   do k=0, nblk0(3)-1
      do j=0, nblk0(2)-1
         do i=0, nblk0(1)-1
            if (amr .and. all([i, j, k] == refine)) then
               do o=0, 7
                  if (extra .and. o == 7) then
                     do o2=0, 7
                        g = g + 1
                        leaves(:,g) = [4*i + 2 + ibits(o2, 0, 1), 4*j + 2 + ibits(o2, 1, 1), 4*k + 2 + ibits(o2, 2, 1), &
                                       2_I4P]
                     enddo
                  else
                     g = g + 1
                     leaves(:,g) = [2*i + ibits(o, 0, 1), 2*j + ibits(o, 1, 1), 2*k + ibits(o, 2, 1), 1_I4P]
                  endif
               enddo
            else
               g = g + 1
               leaves(:,g) = [i, j, k, 0_I4P]
            endif
         enddo
      enddo
   enddo
   nb = 0
   do g=0, nleaf-1
      if (mod(g, wsize) == wrank) nb = nb + 1
   enddo
   allocate(coords(4,nb))
   b = 0
   do g=0, nleaf-1
      if (mod(g, wsize) /= wrank) cycle
      b = b + 1
      coords(:,b) = leaves(:,g+1)
   enddo
   call mesh%init(NIJK, NGHOST, nblk0, dx0, [0._R_P, 0._R_P, 0._R_P], [periodic, periodic, periodic], &
                  [.false., .false., .false.], coords, stat, msg)
   endsubroutine make_mesh

   subroutine run_case(label, nblk0, dx0, amr, refine, root_cells, periodic, isolated)
   !< One mesh: the setup, the solve and its checks by meaning.
   character(*), intent(in) :: label      !< Case label.
   integer(I4P), intent(in) :: nblk0(3)   !< Blocks per axis at level 0.
   real(R_P),    intent(in) :: dx0(3)     !< Cell spacing at level 0.
   logical,      intent(in) :: amr        !< Refine block `refine` to level 1.
   integer(I4P), intent(in) :: refine(3)  !< The refined level-0 block (0-based).
   integer(I4P), intent(in) :: root_cells !< Most cells of the root (0: the default).
   logical,      intent(in) :: periodic   !< Periodic axes (all or none).
   logical,      intent(in) :: isolated   !< Isolated boundaries.
   type(elisha_mesh_object)                :: mesh   !< Mesh.
   type(elisha_hierarchical_solver_object) :: solver !< Solver.
   real(R_P), allocatable :: buf(:), phi(:,:,:,:), au(:), x(:), y(:) !< Source, potential, A u, symmetry vectors.
   real(R_P)              :: dx(3), xc(3), vol, s(2), mean, res(2), sym, length(3) !< Measures.
   real(R_P)              :: vsym, vsym_adam  !< The V-cycle's symmetry defect, default and ADAM's.
   integer(I4P)           :: b, i, j, k, n, iterations, nleaf(1), it_adam !< Counters.
   character(:), allocatable :: adam_note     !< ADAM's V-cycle on AMR (printed).

   call make_mesh(mesh, nblk0, dx0, amr, refine, periodic, .false., stat, msg)
   if (stat /= ELISHA_OK) then
      call fail(label//': mesh init: '//msg)
      return
   endif
   if (root_cells > 0) then
      call solver%setup(mesh, c, stat, msg, tolerance=TOL_MG, max_iterations=MAX_IT, root_cells=root_cells, &
                        isolated=isolated)
   else
      call solver%setup(mesh, c, stat, msg, tolerance=TOL_MG, max_iterations=MAX_IT, isolated=isolated)
   endif
   if (stat /= ELISHA_OK) then
      call fail(label//': setup: '//msg)
      return
   endif
   length = nblk0 * NIJK * dx0
   allocate(buf(max(mesh%nb*product(NIJK),1)), au(max(mesh%nb*product(NIJK),1)))
   allocate(x(max(mesh%nb*product(NIJK),1)), y(max(mesh%nb*product(NIJK),1)))
   allocate(phi(1-NGHOST:NIJK(1)+NGHOST,1-NGHOST:NIJK(2)+NGHOST,1-NGHOST:NIJK(3)+NGHOST,1:max(mesh%nb,1)))
   phi = 0._R_P ; buf = 0._R_P ; au = 0._R_P
   ! the source at the cell centres; the volume-weighted mean as C0 plus the mean deviation from it
   s = 0._R_P ; n = 0
   do b=1, mesh%nb
      dx = mesh%cell_dx(mesh%coords(4,b))
      vol = product(dx)
      do k=1, NIJK(3)
         do j=1, NIJK(2)
            do i=1, NIJK(1)
               xc = mesh%emin + mesh%coords(1:3,b) * NIJK * dx + ([i, j, k] - 0.5_R_P) * dx
               n = n + 1
               buf(n) = source(xc, length)
               s = s + [vol * (buf(n) - C0), vol]
            enddo
         enddo
      enddo
   enddo
   call c%sum_r(s, stat)
   mean = C0 + s(1) / s(2)
   if (.not.solver%singular) mean = 0._R_P
   call solver%solve_packed(mesh, buf, FACTOR, phi, stat, msg)
   if (stat /= ELISHA_OK) then
      call fail(label//': solve: '//msg)
      return
   endif
   iterations = solver%iterations
   res = 0._R_P
   if (.not.isolated) then ! the residual recomputed through the operator (isolated: A phi carries -B g)
      n = 0
      do b=1, mesh%nb
         do k=1, NIJK(3)
            do j=1, NIJK(2)
               do i=1, NIJK(1)
                  n = n + 1
                  x(n) = phi(i,j,k,b)
               enddo
            enddo
         enddo
      enddo
      call solver%apply_a(x, au, stat) ; if (stat /= ELISHA_OK) call fail(label//': apply_a')
      n = 0
      do b=1, mesh%nb
         vol = product(mesh%cell_dx(mesh%coords(4,b)))
         do k=1, NIJK(3)
            do j=1, NIJK(2)
               do i=1, NIJK(1)
                  n = n + 1
                  res(1) = max(res(1), abs(au(n) / vol - FACTOR * (buf(n) - mean)))
                  res(2) = max(res(2), abs(FACTOR * (buf(n) - mean)))
               enddo
            enddo
         enddo
      enddo
      call c%max_real8(res, stat)
   endif
   do i=1, size(x)
      x(i) = sin(1.3_R_P * i + 0.7_R_P * wrank) ; y(i) = cos(2.1_R_P * i - 0.4_R_P * wrank)
   enddo
   sym = solver%symmetry_defect(x, y, stat) ; if (stat /= ELISHA_OK) call fail(label//': symmetry_defect')
   vsym = solver%vcycle_symmetry_defect(x, y, stat) ; if (stat /= ELISHA_OK) call fail(label//': vcycle_symmetry_defect')
   adam_note = ''
   vsym_adam = huge(1._R_P)
   if (amr) call adam_vcycle(label, mesh, root_cells, isolated, buf, phi, x, y, it_adam, vsym_adam, adam_note)
   nleaf = mesh%nleaf
   if (wrank == 0) print '(A,I0,A,I0,A,I0,A,I0,A,I0,A,I0,A,ES9.2,A,ES9.2,A,ES9.2,A,ES9.2,A,ES9.2,A)', &
      'elisha_test_hierarchical_solver_mpi: '//label//' np ', wsize, ' leaves ', nleaf(1), ' levels ', solver%lbot, &
      '..', solver%lmax, ' iterations ', solver%iterations0, '+', iterations, ' residual ', solver%residual, ' tol ', &
      solver%tolerance_used, ' recomputed ', res(1) / max(res(2), tiny(1._R_P)), ' symmetry ', sym, &
      ' V-cycle symmetry ', vsym, adam_note
   if (.not.(solver%residual <= solver%tolerance_used)) call fail(label//': FCG residual above its stop tolerance')
   if (.not.(iterations > 0)) call fail(label//': no FCG iteration')
   if (.not.(res(1) <= 2._R_P * solver%tolerance_used * res(2))) call fail(label//': recomputed residual above 2 tol')
   if (.not.(sym <= TOL_SYMMETRY)) call fail(label//': symmetry defect above 1e-12')
   if (.not.(vsym <= TOL_SYMMETRY)) call fail(label//': V-cycle symmetry defect above 1e-12')
   if (amr .and. .not.(vsym_adam >= MIN_ADAM_DEFECT)) call fail(label//': ADAM''s V-cycle symmetric on AMR (below 1e-6)')
   if (label(1:3) == '(c)' .and. iterations /= 1) call fail(label//': the exact V-cycle took more than one iteration')
   call solver%destroy
   call mesh%destroy
   endsubroutine run_case

   subroutine adam_vcycle(label, mesh, root_cells, isolated, buf, phi, x, y, it, defect, note)
   !< ADAM's V-cycle (`ELISHA_VCYCLE_ADAM`) on a case's mesh: the same solve, its iterations and its V-cycle's symmetry
   !< defect for the same pair of vectors.
   character(*),             intent(in)    :: label      !< Case label.
   type(elisha_mesh_object), intent(inout) :: mesh       !< The case's mesh.
   integer(I4P),             intent(in)    :: root_cells !< Most cells of the root (0: the default).
   logical,                  intent(in)    :: isolated   !< Isolated boundaries.
   real(R_P),                intent(in)    :: buf(:)     !< Source.
   real(R_P),                intent(in)    :: phi(:,:,:,:) !< The case's potential (its shape).
   real(R_P),                intent(in)    :: x(:), y(:) !< The symmetry vectors.
   integer(I4P),             intent(out)   :: it         !< FCG iterations.
   real(R_P),                intent(out)   :: defect     !< The V-cycle's symmetry defect.
   character(:), allocatable, intent(out)  :: note       !< Printed.
   type(elisha_hierarchical_solver_object) :: adam       !< ADAM's V-cycle.
   real(R_P), allocatable :: phi_a(:,:,:,:)              !< Its potential.

   it = -1 ; defect = huge(1._R_P) ; note = ''
   if (root_cells > 0) then
      call adam%setup(mesh, c, stat, msg, tolerance=TOL_MG, max_iterations=MAX_IT, root_cells=root_cells, &
                      isolated=isolated, vcycle=ELISHA_VCYCLE_ADAM)
   else
      call adam%setup(mesh, c, stat, msg, tolerance=TOL_MG, max_iterations=MAX_IT, isolated=isolated, &
                      vcycle=ELISHA_VCYCLE_ADAM)
   endif
   if (stat /= ELISHA_OK) then
      call fail(label//': ADAM V-cycle setup: '//msg)
      return
   endif
   allocate(phi_a(1-NGHOST:NIJK(1)+NGHOST,1-NGHOST:NIJK(2)+NGHOST,1-NGHOST:NIJK(3)+NGHOST,1:size(phi, dim=4)))
   phi_a = 0._R_P
   call adam%solve_packed(mesh, buf, FACTOR, phi_a, stat, msg)
   if (stat /= ELISHA_OK) then
      call fail(label//': ADAM V-cycle solve: '//msg)
      return
   endif
   it = adam%iterations
   defect = adam%vcycle_symmetry_defect(x, y, stat)
   if (stat /= ELISHA_OK) call fail(label//': ADAM V-cycle vcycle_symmetry_defect')
   note = ', ADAM''s V-cycle: iterations '//itoa(it)//', V-cycle symmetry '//trim(adjustl(fmt(defect)))
   call adam%destroy
   endsubroutine adam_vcycle

   subroutine check_refusals
   !< The refusals, each with the same status on every rank.
   type(elisha_mesh_object)                :: mesh   !< Mesh.
   type(elisha_hierarchical_solver_object) :: solver !< Solver.
   real(R_P), allocatable :: buf(:), phi(:,:,:,:), ghost(:) !< Source, potential, boundary values.
   integer(I4P)           :: bc(2,3)                 !< Kinds.

   ! not 2:1: a level-2 corner next to level-0 leaves
   call make_mesh(mesh, [2_I4P, 2_I4P, 2_I4P], [1._R_P, 1._R_P, 1._R_P] / 16._R_P, .true., [0_I4P, 0_I4P, 0_I4P], &
                  .true., .true., stat, msg)
   if (stat /= ELISHA_OK) call fail('refusal mesh init: '//msg)
   call solver%setup(mesh, c, stat, msg)
   call expect(stat, ELISHA_ERR_MESH_BALANCE, 'a mesh not 2:1 balanced')
   call mesh%destroy
   ! isolated with boundary kinds
   call make_mesh(mesh, [2_I4P, 2_I4P, 2_I4P], [1._R_P, 1._R_P, 1._R_P] / 16._R_P, .false., [0_I4P, 0_I4P, 0_I4P], &
                  .false., .false., stat, msg)
   bc = ELISHA_BC_DIRICHLET_GHOST
   call solver%setup(mesh, c, stat, msg, isolated=.true., bc=bc)
   call expect(stat, ELISHA_ERR_ISOLATED_BC, 'isolated with bc')
   ! a vcycle that is no ELISHA_VCYCLE_*
   call solver%setup(mesh, c, stat, msg, vcycle=7_I4P)
   call expect(stat, ELISHA_ERR_ARGUMENT, 'a vcycle that is no ELISHA_VCYCLE_*')
   ! the device assembly on the host solver
   call solver%setup(mesh, c, stat, msg, assembly=ELISHA_ASSEMBLY_DEVICE)
   call expect(stat, ELISHA_ERR_ARGUMENT, 'the device assembly on the host solver')
   ! a solve before the setup (the failed setups left the solver destroyed)
   allocate(buf(max(mesh%nb*product(NIJK),1)), ghost(1))
   allocate(phi(1-NGHOST:NIJK(1)+NGHOST,1-NGHOST:NIJK(2)+NGHOST,1-NGHOST:NIJK(3)+NGHOST,1:max(mesh%nb,1)))
   buf = 1._R_P ; phi = 0._R_P ; ghost = 0._R_P
   call solver%solve_packed(mesh, buf, FACTOR, phi, stat, msg)
   call expect(stat, ELISHA_ERR_NOT_INITIALIZED, 'solve before setup')
   ! the ghost-Dirichlet (default) setup: a wrong phi shape, then fewer ghost values than boundary faces
   call solver%setup(mesh, c, stat, msg)
   call expect(stat, ELISHA_OK, 'bounded setup')
   if (stat == ELISHA_OK) then
      deallocate(phi)
      allocate(phi(1-NGHOST:NIJK(1)+NGHOST-1,1-NGHOST:NIJK(2)+NGHOST,1-NGHOST:NIJK(3)+NGHOST,1:max(mesh%nb,1)))
      call solver%solve_packed(mesh, buf, FACTOR, phi, stat, msg)
      call expect(stat, ELISHA_ERR_SHAPE, 'a phi of the wrong shape')
      deallocate(phi)
      allocate(phi(1-NGHOST:NIJK(1)+NGHOST,1-NGHOST:NIJK(2)+NGHOST,1-NGHOST:NIJK(3)+NGHOST,1:max(mesh%nb,1)))
      call solver%solve_packed(mesh, buf, FACTOR, phi, stat, msg, ghost=ghost)
      call expect(stat, ELISHA_ERR_GHOST_COUNT, 'fewer ghost values than faces')
   endif
   call solver%destroy
   call mesh%destroy
   endsubroutine check_refusals

   subroutine check_not_converged(periodic)
   !< ELISHA notes E1c, decision C: an FCG stopped at max_iterations above its tolerance is ELISHA_ERR_NOT_CONVERGED on
   !< every rank, phi the last iterate (finite everywhere, ghosts included); the next solve with enough iterations
   !< converges. The AMR corner mesh, periodic, or isolated (the James step: its two solves).
   logical, intent(in)                     :: periodic !< Periodic, else isolated.
   type(elisha_mesh_object)                :: mesh     !< Mesh.
   type(elisha_hierarchical_solver_object) :: solver   !< Solver.
   real(R_P), allocatable :: buf(:), phi(:,:,:,:)      !< Source, potential.
   real(R_P)              :: amax(1)                   !< Max |phi| over the interiors and the ranks.
   logical                :: finite(1)                 !< phi finite everywhere (ghosts included), over the ranks.
   integer(I4P)           :: b, i, n, nan_(1)          !< Counters.
   character(:), allocatable :: what                   !< Label.

   what = merge('periodic', 'isolated', periodic)
   ! periodic: the AMR corner mesh; isolated (every boundary leaf at the coarsest level): uniform, the root capped at
   ! 64 cells (levels inside the blocks: no exact V-cycle)
   call make_mesh(mesh, [2_I4P, 2_I4P, 2_I4P], [1._R_P, 1._R_P, 1._R_P] / 16._R_P, periodic, [0_I4P, 0_I4P, 0_I4P], &
                  periodic, .false., stat, msg)
   if (stat /= ELISHA_OK) then
      call fail('not converged, '//what//': mesh init: '//msg)
      return
   endif
   allocate(buf(max(mesh%nb*product(NIJK),1)))
   allocate(phi(1-NGHOST:NIJK(1)+NGHOST,1-NGHOST:NIJK(2)+NGHOST,1-NGHOST:NIJK(3)+NGHOST,1:max(mesh%nb,1)))
   n = 0
   do b=1, mesh%nb
      do i=1, product(NIJK)
         n = n + 1
         buf(n) = C0 + sin(0.37_R_P * n + 1.1_R_P * wrank) ! no smooth structure: one V-cycle cannot solve it
      enddo
   enddo
   phi = 0._R_P
   call solver%setup(mesh, c, stat, msg, tolerance=TOL_MG, max_iterations=1_I4P, isolated=.not.periodic, &
                     root_cells=64_I4P)
   if (stat /= ELISHA_OK) then
      call fail('not converged, '//what//': setup: '//msg)
      return
   endif
   call solver%solve_packed(mesh, buf, FACTOR, phi, stat, msg)
   call expect(stat, ELISHA_ERR_NOT_CONVERGED, 'an FCG stopped at max_iterations = 1, '//what)
   if (solver%converged) call fail('not converged, '//what//': converged is true')
   finite = .true.
   amax = 0._R_P
   do b=1, mesh%nb
      if (any(phi(:,:,:,b) /= phi(:,:,:,b)) .or. any(abs(phi(:,:,:,b)) > huge(1._R_P))) finite = .false.
      amax = max(amax(1), maxval(abs(phi(1:NIJK(1),1:NIJK(2),1:NIJK(3),b))))
   enddo
   nan_ = merge(0_I4P, 1_I4P, finite(1))
   call c%max_i4(nan_, stat)
   call c%max_real8(amax, stat)
   if (nan_(1) /= 0) call fail('not converged, '//what//': phi not finite')
   if (.not.(amax(1) > 0._R_P)) call fail('not converged, '//what//': phi is zero (no last iterate)')
   ! the same solver, enough iterations: a new setup (max_iterations is a setup argument)
   call solver%setup(mesh, c, stat, msg, tolerance=TOL_MG, max_iterations=MAX_IT, isolated=.not.periodic, &
                     root_cells=64_I4P)
   if (stat == ELISHA_OK) call solver%solve_packed(mesh, buf, FACTOR, phi, stat, msg)
   call expect(stat, ELISHA_OK, 'the same solve with max_iterations = '//itoa(MAX_IT)//', '//what)
   if (.not.solver%converged) call fail('not converged, '//what//': converged is false after a converged solve')
   call solver%destroy
   call mesh%destroy
   endsubroutine check_not_converged

   subroutine expect(got, want, what)
   !< A status against the expected one.
   integer(I4P), intent(in) :: got, want !< Statuses.
   character(*), intent(in) :: what      !< The refusal.

   character(:), allocatable :: why      !< The message, when there is one.

   why = ''
   if (allocated(msg)) why = ' ('//msg//')'
   if (got /= want) then
      call fail(what//': status '//itoa(got)//', expected '//itoa(want)//why)
   elseif (wrank == 0) then
      print '(A)', 'elisha_test_hierarchical_solver_mpi: as expected: '//what
   endif
   endsubroutine expect

   pure function source(xc, length) result(f)
   !< The source at the point xc of a box of sides `length` (lower corner at the origin): a constant plus smooth
   !< periodic modes.
   real(R_P), intent(in) :: xc(3)     !< Point.
   real(R_P), intent(in) :: length(3) !< Box sides.
   real(R_P)             :: f         !< Source.
   real(R_P)             :: t(3)      !< Phases per unit wavenumber.

   t = 2._R_P * PI * xc / length
   f = C0 + cos(t(1)) + 0.5_R_P * cos(2._R_P * t(2) + t(3) + 0.3_R_P) &
       + 0.25_R_P * sin(3._R_P * t(1) + t(2) + 2._R_P * t(3))
   endfunction source

   pure function itoa(i) result(s)
   !< An integer as text.
   integer(I4P), intent(in)  :: i !< Integer.
   character(:), allocatable :: s !< Text.
   character(16)             :: w !< Buffer.

   write(w, '(I0)') i
   s = trim(w)
   endfunction itoa

   pure function fmt(x) result(s)
   !< A real as text (ES9.2).
   real(R_P), intent(in)     :: x !< Real.
   character(:), allocatable :: s !< Text.
   character(16)             :: w !< Buffer.

   write(w, '(ES9.2)') x
   s = trim(w)
   endfunction fmt

   subroutine fail(what)
   !< Report a failed check.
   character(*), intent(in) :: what !< What failed.

   failures = failures + 1
   print '(A,I0,A)', '  FAIL rank ', wrank, ': '//what
   endsubroutine fail
endprogram elisha_test_hierarchical_solver_mpi
