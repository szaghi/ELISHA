!< ELISHA test: the facade chooses the path, forwards bitwise, reports its diagnostics and refuses.
program elisha_test_solver_mpi
!< ELISHA test: the facade chooses the path, forwards bitwise, reports its diagnostics and refuses.
!<
!< 3-D meshes of 8^3-cell blocks with 3 ghost layers, the leaves dealt round-robin to the ranks, the source a constant
!< plus smooth modes at the cell centres. Each case sets up the facade (`elisha_solver_object`) and, beside it, the
!< solver the path names, called directly with the same arguments (ADAM's facade call sequence), solves with both and
!< checks: the path chosen (`path()`, `is_hierarchical()`, `diagnostics()%path`); `phi` bitwise equal over every cell,
!< ghosts included; the energy bitwise equal (the spectral solver's times the cell volume, as ADAM's facade); the
!< diagnostics equal to the solver's own fields (iterations, iterations0, residual, tolerance_used, pcg), the wall
!< times not negative (they are measurements: no equality), and `reset_wall` zeroing them. Cases:
!< - (a) uniform periodic, auto: spectral;
!< - (b) AMR periodic (a corner block refined), auto: hierarchical;
!< - (c) uniform periodic, hierarchical forced, the root capped at 64 cells;
!< - (d) uniform isolated, auto: spectral (Hockney-Eastwood);
!< - (e) uniform isolated, hierarchical forced: the James step (iterations0 > 0);
!< - (f) uniform, zero-ghost faces on x and the wide D(G) of order 4, spectral: the CG path (pcg; iterations on rank 0);
!< - (g) AMR bounded, auto (ghost-Dirichlet faces), with boundary values on the faces `boundary_faces` lists;
!< - (h), (i) AMR and uniform periodic, auto, both molds given (the host types): the solver's type, and on the
!<   hierarchical path the spectral mold forwarded as `coarse`.
!< Refusals, one check each, each with the same status on every rank: a spectral path on an AMR mesh
!< (`ELISHA_ERR_MESH_LEVELS`), boundary values on the spectral path (`ELISHA_ERR_GHOST_VALUES`), `boundary_faces` on it
!< (the same), isolated with `bc` (`ELISHA_ERR_ISOLATED_BC`), an unknown path (`ELISHA_ERR_ARGUMENT`), the wide D(G) on
!< the hierarchical path (`ELISHA_ERR_DG_AMR`), a context not initialised, a solve and `boundary_faces` before the setup
!< and after a failed one (`ELISHA_ERR_NOT_INITIALIZED`, `path()` 0), `on_device` on the host solvers of both paths
!< (`ELISHA_ERR_ARGUMENT`); and an FCG stopped at `max_iterations` = 1 on the AMR mesh (`ELISHA_ERR_NOT_CONVERGED`,
!< `diagnostics()%converged` false, the energy set and finite; then converged with 200). The FP traps of a debug build are off around
!< `MPI_Init` only (hwloc raises IEEE flags inside it).
#ifndef ELISHA_NO_MPI
use mpi
#endif
use, intrinsic :: ieee_exceptions, only : ieee_usual, ieee_get_halting_mode, ieee_set_halting_mode
use elisha, only : I4P, R_P, ELISHA_OK, ELISHA_ERR_MESH_LEVELS, ELISHA_ERR_GHOST_VALUES, ELISHA_ERR_ISOLATED_BC,      &
                   ELISHA_ERR_ARGUMENT, ELISHA_ERR_DG_AMR, ELISHA_ERR_NOT_INITIALIZED, ELISHA_BC_PERIODIC,           &
                   ELISHA_BC_ZERO_GHOSTS, ELISHA_BC_DIRICHLET_GHOST, ELISHA_PATH_AUTO, ELISHA_PATH_SPECTRAL,         &
                   ELISHA_PATH_HIERARCHICAL, elisha_comm_object, elisha_mesh_object, elisha_spectral_solver_object, &
                   elisha_hierarchical_solver_object, elisha_solver_object, elisha_diagnostics_object, &
                   ELISHA_ERR_NOT_CONVERGED
implicit none
real(R_P),    parameter  :: TOL_MG=1.e-11_R_P       !< FCG tolerance requested.
integer(I4P), parameter  :: MAX_IT=100_I4P          !< FCG iterations allowed.
real(R_P),    parameter  :: FACTOR=2.5_R_P          !< Source factor.
real(R_P),    parameter  :: PI=acos(-1._R_P)        !< Pi.
real(R_P),    parameter  :: C0=0.7_R_P              !< The constant of the source.
integer(I4P), parameter  :: NGHOST=3_I4P            !< Ghost layers.
integer(I4P), parameter  :: NIJK(3)=[8_I4P, 8_I4P, 8_I4P] !< Cells per block per axis.
integer(I4P), parameter  :: N2(3)=[2_I4P, 2_I4P, 2_I4P]   !< Blocks per axis at level 0.
real(R_P),    parameter  :: DX(3)=[1._R_P, 1._R_P, 1._R_P] / 16._R_P !< Cell spacing at level 0.
logical,      parameter  :: T_=.true., F_=.false.   !< Shorthands.
type(elisha_comm_object) :: c                       !< Communication context.
integer(I4P)             :: failures, nfail(1), stat !< Failed checks, over the ranks; status.
integer                  :: wrank, wsize, world, ierr !< MPI.
logical                  :: halting(3)              !< The FP halting modes of the build (debug: on).
character(:), allocatable :: msg                    !< Message.
integer(I4P)             :: bcz(2,3)                !< Zero-ghost faces on x, periodic elsewhere.

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
bcz = ELISHA_BC_PERIODIC ; bcz(:,1) = ELISHA_BC_ZERO_GHOSTS
call run_case('(a) uniform, auto        ', amr=F_, periodic=[T_,T_,T_], path=ELISHA_PATH_AUTO, &
              want=ELISHA_PATH_SPECTRAL)
call run_case('(b) AMR, auto            ', amr=T_, periodic=[T_,T_,T_], path=ELISHA_PATH_AUTO, &
              want=ELISHA_PATH_HIERARCHICAL)
call run_case('(c) uniform, hierarchical', amr=F_, periodic=[T_,T_,T_], path=ELISHA_PATH_HIERARCHICAL, &
              want=ELISHA_PATH_HIERARCHICAL, root_cells=64_I4P)
call run_case('(d) isolated, auto       ', amr=F_, periodic=[F_,F_,F_], path=ELISHA_PATH_AUTO, &
              want=ELISHA_PATH_SPECTRAL, isolated=T_)
call run_case('(e) isolated, hierarch.  ', amr=F_, periodic=[F_,F_,F_], path=ELISHA_PATH_HIERARCHICAL, &
              want=ELISHA_PATH_HIERARCHICAL, isolated=T_)
call run_case('(f) zero ghosts D(G) p=4 ', amr=F_, periodic=[F_,T_,T_], path=ELISHA_PATH_SPECTRAL, &
              want=ELISHA_PATH_SPECTRAL, bc=bcz, dg_order=4_I4P)
call run_case('(g) AMR bounded, ghosts  ', amr=T_, periodic=[F_,F_,F_], path=ELISHA_PATH_AUTO, &
              want=ELISHA_PATH_HIERARCHICAL, with_ghost=T_)
call run_case('(h) AMR, molds           ', amr=T_, periodic=[T_,T_,T_], path=ELISHA_PATH_AUTO, &
              want=ELISHA_PATH_HIERARCHICAL, molds=T_)
call run_case('(i) uniform, molds       ', amr=F_, periodic=[T_,T_,T_], path=ELISHA_PATH_AUTO, &
              want=ELISHA_PATH_SPECTRAL, molds=T_)
call check_refusals
nfail = failures
call c%max_i4(nfail, stat)
call c%destroy(stat)
#ifndef ELISHA_NO_MPI
call MPI_FINALIZE(ierr)
#endif
if (nfail(1) > 0) error stop 1
if (wrank == 0) print '(A,I0,A)', 'elisha_test_solver_mpi: ok (', wsize, ' rank(s))'

contains
   subroutine make_mesh(mesh, amr, periodic)
   !< The mesh: 2x2x2 level-0 blocks of NIJK cells, block (0,0,0) refined to level 1 when `amr`; leaves round-robin.
   type(elisha_mesh_object), intent(inout) :: mesh        !< Mesh.
   logical,                  intent(in)    :: amr         !< Refine block (0,0,0).
   logical,                  intent(in)    :: periodic(3) !< Periodic axes.
   integer(I4P), allocatable :: leaves(:,:), coords(:,:)  !< Every leaf, this rank's leaves [4,*].
   integer(I4P)              :: nleaf, g, i, j, k, o, nb, b !< Counters.

   nleaf = product(N2) + merge(7_I4P, 0_I4P, amr)
   allocate(leaves(4,nleaf))
   g = 0
   do k=0, N2(3)-1
      do j=0, N2(2)-1
         do i=0, N2(1)-1
            if (amr .and. all([i, j, k] == 0)) then
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
   call mesh%init(NIJK, NGHOST, N2, DX, [0._R_P, 0._R_P, 0._R_P], periodic, [F_, F_, F_], coords, stat, msg)
   if (stat /= ELISHA_OK) call fail('mesh init: '//msg)
   endsubroutine make_mesh

   subroutine fill_source(mesh, buf)
   !< The source at the cell centres of this rank's blocks: a constant plus smooth periodic modes.
   type(elisha_mesh_object), intent(in)  :: mesh    !< Mesh.
   real(R_P), allocatable,   intent(out) :: buf(:)  !< Packed source.
   real(R_P)                             :: h(3), xc(3), t(3), length(3) !< Spacing, centre, phases, domain.
   integer(I4P)                          :: b, i, j, k, n !< Counters.

   allocate(buf(max(mesh%nb*product(NIJK),1)))
   buf = 0._R_P
   length = N2 * NIJK * DX
   n = 0
   do b=1, mesh%nb
      h = mesh%cell_dx(mesh%coords(4,b))
      do k=1, NIJK(3)
         do j=1, NIJK(2)
            do i=1, NIJK(1)
               xc = mesh%coords(1:3,b) * NIJK * h + ([i, j, k] - 0.5_R_P) * h
               t = 2._R_P * PI * xc / length
               n = n + 1
               buf(n) = C0 + cos(t(1)) + 0.5_R_P * cos(2._R_P * t(2) + t(3) + 0.3_R_P) &
                        + 0.25_R_P * sin(3._R_P * t(1) + t(2) + 2._R_P * t(3))
            enddo
         enddo
      enddo
   enddo
   endsubroutine fill_source

   subroutine run_case(label, amr, periodic, path, want, isolated, bc, dg_order, root_cells, with_ghost, molds)
   !< One case: the facade and the solver of its path, the same arguments, the same results (see the program).
   character(*),           intent(in) :: label       !< Case label.
   logical,                intent(in) :: amr         !< AMR mesh.
   logical,                intent(in) :: periodic(3) !< Periodic axes.
   integer(I4P),           intent(in) :: path        !< The path asked.
   integer(I4P),           intent(in) :: want        !< The path expected.
   logical,      optional, intent(in) :: isolated    !< Isolated boundaries.
   integer(I4P), optional, intent(in) :: bc(2,3)     !< Kinds.
   integer(I4P), optional, intent(in) :: dg_order    !< Operator order.
   integer(I4P), optional, intent(in) :: root_cells  !< Most cells of the root (hierarchical).
   logical,      optional, intent(in) :: with_ghost  !< Boundary values on the faces.
   logical,      optional, intent(in) :: molds       !< Give both molds (the host types).
   type(elisha_mesh_object)                :: mesh          !< Mesh.
   type(elisha_solver_object)              :: facade        !< The facade.
   type(elisha_spectral_solver_object)     :: sp, sp_mold   !< The spectral solver, direct; a mold.
   type(elisha_hierarchical_solver_object) :: hi, hi_mold   !< The hierarchical solver, direct; a mold.
   type(elisha_diagnostics_object)         :: d             !< Diagnostics.
   real(R_P), allocatable :: buf(:), phi_f(:,:,:,:), phi_d(:,:,:,:), ghost(:), x(:,:) !< Source, potentials, values.
   integer(I4P), allocatable :: kinds(:), dir(:)            !< Boundary faces.
   real(R_P)              :: e_f, e_d                       !< Energies.
   integer(I4P)           :: n, ndiff(1)                    !< Counters.
   logical                :: ghosts, use_molds              !< Options.

   ghosts = .false. ; if (present(with_ghost)) ghosts = with_ghost
   use_molds = .false. ; if (present(molds)) use_molds = molds
   call make_mesh(mesh, amr, periodic)
   if (use_molds) then
      call facade%setup(mesh, c, stat, msg, path=path, isolated=isolated, bc=bc, dg_order=dg_order, tolerance=TOL_MG, &
                        max_iterations=MAX_IT, root_cells=root_cells, spectral_mold=sp_mold, hierarchical_mold=hi_mold)
   else
      call facade%setup(mesh, c, stat, msg, path=path, isolated=isolated, bc=bc, dg_order=dg_order, tolerance=TOL_MG, &
                        max_iterations=MAX_IT, root_cells=root_cells)
   endif
   if (stat /= ELISHA_OK) then
      call fail(label//': facade setup: '//msg)
      return
   endif
   if (facade%path() /= want) call fail(label//': path '//itoa(facade%path())//', expected '//itoa(want))
   if (facade%is_hierarchical() .neqv. (want == ELISHA_PATH_HIERARCHICAL)) call fail(label//': is_hierarchical')
   ! the solver of the path, directly, with the facade's call sequence
   if (want == ELISHA_PATH_SPECTRAL) then
      call sp%setup(mesh=mesh, comm=c, stat=stat, errmsg=msg, isolated=isolated, bc=bc, dg_order=dg_order)
   elseif (use_molds) then
      call hi%setup(mesh=mesh, comm=c, stat=stat, errmsg=msg, tolerance=TOL_MG, max_iterations=MAX_IT, &
                    coarse=sp_mold, isolated=isolated, bc=bc, root_cells=root_cells)
   else
      call hi%setup(mesh=mesh, comm=c, stat=stat, errmsg=msg, tolerance=TOL_MG, max_iterations=MAX_IT, &
                    isolated=isolated, bc=bc, root_cells=root_cells)
   endif
   if (stat /= ELISHA_OK) then
      call fail(label//': direct setup: '//msg)
      return
   endif
   call fill_source(mesh, buf)
   allocate(phi_f(1-NGHOST:NIJK(1)+NGHOST,1-NGHOST:NIJK(2)+NGHOST,1-NGHOST:NIJK(3)+NGHOST,1:max(mesh%nb,1)))
   allocate(phi_d, mold=phi_f)
   phi_f = 0._R_P ; phi_d = 0._R_P
   if (ghosts) then ! the values at the faces boundary_faces lists, a linear function of where they live
      call facade%boundary_faces(kinds, dir, x, stat, msg)
      if (stat /= ELISHA_OK) then
         call fail(label//': boundary_faces: '//msg)
         return
      endif
      if (size(kinds) /= hi%nbnd .or. size(kinds) /= size(dir) .or. size(x, dim=2) /= size(kinds)) &
         call fail(label//': boundary_faces size')
      if (size(kinds) == hi%nbnd) then
         if (any(kinds /= hi%bnd_kind(1:hi%nbnd)) .or. any(dir /= hi%bnd_dir(1:hi%nbnd)) .or. &
             any(x /= hi%bnd_x(:,1:hi%nbnd))) call fail(label//': boundary_faces differ from the solver''s')
      endif
      allocate(ghost(max(size(kinds),1)))
      ghost = 0._R_P
      do n=1, size(kinds)
         ghost(n) = 0.3_R_P + 0.1_R_P * x(1,n) - 0.2_R_P * x(2,n) + 0.05_R_P * x(3,n)
      enddo
      call facade%solve_packed(mesh, buf, FACTOR, phi_f, stat, msg, energy=e_f, ghost=ghost)
   else
      call facade%solve_packed(mesh, buf, FACTOR, phi_f, stat, msg, energy=e_f)
   endif
   if (stat /= ELISHA_OK) then
      call fail(label//': facade solve: '//msg)
      return
   endif
   if (want == ELISHA_PATH_SPECTRAL) then
      call sp%solve_packed(mesh=mesh, buf=buf, factor=FACTOR, phi=phi_d, stat=stat, errmsg=msg, energy=e_d)
      e_d = e_d * product(sp%dxyz)
   elseif (ghosts) then
      call hi%solve_packed(mesh=mesh, buf=buf, factor=FACTOR, phi=phi_d, stat=stat, errmsg=msg, energy=e_d, ghost=ghost)
   else
      call hi%solve_packed(mesh=mesh, buf=buf, factor=FACTOR, phi=phi_d, stat=stat, errmsg=msg, energy=e_d)
   endif
   if (stat /= ELISHA_OK) then
      call fail(label//': direct solve: '//msg)
      return
   endif
   ! bitwise: phi on every cell, ghosts included, and the energy
   ndiff = count(phi_f /= phi_d)
   call c%max_i4(ndiff, stat)
   if (ndiff(1) /= 0) call fail(label//': phi differs from the direct solve on '//itoa(ndiff(1))//' cells')
   if (e_f /= e_d) call fail(label//': energy differs from the direct solve')
   ! the diagnostics against the solver's own fields
   d = facade%diagnostics()
   if (d%path /= want) call fail(label//': diagnostics path')
   if (want == ELISHA_PATH_SPECTRAL) then
      if (d%pcg .neqv. sp%pcg) call fail(label//': diagnostics pcg')
      if (d%iterations0 /= 0) call fail(label//': diagnostics iterations0 on the spectral path')
      if (sp%pcg) then
         if (d%iterations /= sp%iterations .or. d%residual /= sp%residual .or. d%tolerance_used /= sp%tolerance) &
            call fail(label//': diagnostics of the CG differ from the solver''s')
         if (wrank == 0 .and. .not.(d%iterations > 0 .and. d%residual <= d%tolerance_used)) &
            call fail(label//': the CG on rank 0 did not iterate to its tolerance')
      elseif (d%iterations /= 0 .or. d%residual /= 0._R_P .or. d%tolerance_used /= 0._R_P) then
         call fail(label//': diagnostics of a direct spectral solve not zero')
      endif
   else
      if (d%pcg) call fail(label//': diagnostics pcg on the hierarchical path')
      if (d%iterations /= hi%iterations .or. d%iterations0 /= hi%iterations0 .or. d%residual /= hi%residual .or. &
          d%tolerance_used /= hi%tolerance_used) call fail(label//': diagnostics differ from the solver''s')
      if (.not.(d%iterations > 0 .and. d%residual <= d%tolerance_used)) &
         call fail(label//': the FCG did not iterate to its tolerance')
      if (present(isolated)) then
         if (isolated .and. .not.(d%iterations0 > 0)) call fail(label//': no zero-ghost solve (James step)')
      endif
   endif
   if (any(d%wall < 0._R_P)) call fail(label//': negative wall time')
   call facade%reset_wall
   d = facade%diagnostics()
   if (any(d%wall /= 0._R_P)) call fail(label//': reset_wall left a wall time')
   if (wrank == 0) print '(A,I0,A,I0,A,I0,A,I0,A,L1,A,ES9.2,A,ES23.15)', 'elisha_test_solver_mpi: '//label//' np ', wsize, &
      ' path ', d%path, ' iterations ', d%iterations0, '+', d%iterations, ' pcg ', d%pcg, ' residual ', d%residual, &
      ' energy ', e_f
   call facade%destroy
   if (facade%path() /= 0) call fail(label//': path after destroy')
   call sp%destroy
   call hi%destroy
   call mesh%destroy
   endsubroutine run_case

   subroutine check_refusals
   !< The refusals, each with the same status on every rank.
   type(elisha_mesh_object)        :: uni, amr   !< A uniform and an AMR mesh.
   type(elisha_solver_object)      :: facade     !< The facade.
   type(elisha_comm_object)        :: none       !< A context never initialised.
   type(elisha_diagnostics_object) :: d          !< Diagnostics.
   real(R_P), allocatable    :: buf(:), phi(:,:,:,:), ghost(:), x(:,:) !< Source, potential, values, faces.
   integer(I4P), allocatable :: kinds(:), dir(:) !< Faces.
   integer(I4P)              :: bc(2,3)          !< Kinds.

   call make_mesh(uni, F_, [T_, T_, T_])
   call make_mesh(amr, T_, [T_, T_, T_])
   allocate(buf(max(uni%nb*product(NIJK),1)), ghost(1))
   allocate(phi(1-NGHOST:NIJK(1)+NGHOST,1-NGHOST:NIJK(2)+NGHOST,1-NGHOST:NIJK(3)+NGHOST,1:max(uni%nb,1)))
   buf = 1._R_P ; phi = 0._R_P ; ghost = 0._R_P
   ! before any setup
   d = facade%diagnostics()
   if (d%path /= 0 .or. facade%path() /= 0 .or. facade%is_hierarchical()) call fail('a path before setup')
   call facade%solve_packed(uni, buf, FACTOR, phi, stat, msg)
   call expect(stat, ELISHA_ERR_NOT_INITIALIZED, 'solve before setup')
   call facade%boundary_faces(kinds, dir, x, stat, msg)
   call expect(stat, ELISHA_ERR_NOT_INITIALIZED, 'boundary_faces before setup')
   call facade%setup(uni, none, stat, msg)
   call expect(stat, ELISHA_ERR_NOT_INITIALIZED, 'a communication context not initialised')
   ! the argument refusals
   call facade%setup(uni, c, stat, msg, path=7_I4P)
   call expect(stat, ELISHA_ERR_ARGUMENT, 'an unknown path')
   bc = ELISHA_BC_DIRICHLET_GHOST
   call facade%setup(uni, c, stat, msg, isolated=.true., bc=bc)
   call expect(stat, ELISHA_ERR_ISOLATED_BC, 'isolated with bc')
   call facade%setup(amr, c, stat, msg, path=ELISHA_PATH_SPECTRAL)
   call expect(stat, ELISHA_ERR_MESH_LEVELS, 'the spectral path on an AMR mesh')
   call facade%setup(amr, c, stat, msg, dg_order=4_I4P)
   call expect(stat, ELISHA_ERR_DG_AMR, 'the wide D(G) on the hierarchical path')
   ! a failed setup leaves the facade not set up
   if (facade%path() /= 0) call fail('a path after a failed setup')
   call facade%solve_packed(uni, buf, FACTOR, phi, stat, msg)
   call expect(stat, ELISHA_ERR_NOT_INITIALIZED, 'solve after a failed setup')
   ! boundary values on the spectral path
   call facade%setup(uni, c, stat, msg)
   call expect(stat, ELISHA_OK, 'uniform setup')
   if (stat == ELISHA_OK) then
      call facade%solve_packed(uni, buf, FACTOR, phi, stat, msg, ghost=ghost)
      call expect(stat, ELISHA_ERR_GHOST_VALUES, 'boundary values on the spectral path')
      call facade%boundary_faces(kinds, dir, x, stat, msg)
      call expect(stat, ELISHA_ERR_GHOST_VALUES, 'boundary_faces on the spectral path')
      call facade%solve_packed(uni, buf, FACTOR, phi, stat, msg, on_device=.true.)
      call expect(stat, ELISHA_ERR_ARGUMENT, 'on_device on the host spectral solver')
   endif
   ! on_device on the host hierarchical solver
   call facade%setup(amr, c, stat, msg)
   call expect(stat, ELISHA_OK, 'AMR setup')
   if (stat == ELISHA_OK) then
      deallocate(buf, phi)
      allocate(buf(max(amr%nb*product(NIJK),1)))
      allocate(phi(1-NGHOST:NIJK(1)+NGHOST,1-NGHOST:NIJK(2)+NGHOST,1-NGHOST:NIJK(3)+NGHOST,1:max(amr%nb,1)))
      buf = 1._R_P ; phi = 0._R_P
      call facade%solve_packed(amr, buf, FACTOR, phi, stat, msg, on_device=.true.)
      call expect(stat, ELISHA_ERR_ARGUMENT, 'on_device on the host hierarchical solver')
   endif
   ! an FCG stopped at max_iterations above its tolerance (ELISHA notes E1c, decision C): a status, the last iterate
   call facade%setup(amr, c, stat, msg, max_iterations=1_I4P)
   call expect(stat, ELISHA_OK, 'AMR setup, max_iterations 1')
   if (stat == ELISHA_OK) then
      block
      real(R_P) :: e !< Energy.
      integer(I4P) :: i !< Counter.
      do i=1, size(buf)
         buf(i) = C0 + sin(0.37_R_P * i + 1.1_R_P * wrank) ! no smooth structure: one V-cycle cannot solve it
      enddo
      e = -huge(1._R_P)
      call facade%solve_packed(amr, buf, FACTOR, phi, stat, msg, energy=e)
      call expect(stat, ELISHA_ERR_NOT_CONVERGED, 'an FCG stopped at max_iterations = 1 (facade)')
      d = facade%diagnostics()
      if (d%converged .or. d%iterations /= 1) call fail('not converged: diagnostics converged '// &
         trim(merge('T', 'F', d%converged))//', iterations '//itoa(d%iterations)//' (expected F, 1)')
      if (e == -huge(1._R_P) .or. e /= e) call fail('not converged: the energy not set')
      call facade%setup(amr, c, stat, msg, max_iterations=200_I4P)
      if (stat == ELISHA_OK) call facade%solve_packed(amr, buf, FACTOR, phi, stat, msg, energy=e)
      call expect(stat, ELISHA_OK, 'the same solve with max_iterations = 200 (facade)')
      d = facade%diagnostics()
      if (.not.d%converged) call fail('converged: diagnostics converged false')
      endblock
   endif
   call facade%destroy
   call uni%destroy
   call amr%destroy
   endsubroutine check_refusals

   subroutine expect(got, want, what)
   !< A status against the expected one.
   integer(I4P), intent(in)  :: got, want !< Statuses.
   character(*), intent(in)  :: what      !< The refusal.
   character(:), allocatable :: why       !< The message, when there is one.

   why = ''
   if (allocated(msg)) why = ' ('//msg//')'
   if (got /= want) then
      call fail(what//': status '//itoa(got)//', expected '//itoa(want)//why)
   elseif (wrank == 0) then
      print '(A)', 'elisha_test_solver_mpi: as expected: '//what//why
   endif
   endsubroutine expect

   pure function itoa(i) result(s)
   !< An integer as text.
   integer(I4P), intent(in)  :: i !< Integer.
   character(:), allocatable :: s !< Text.
   character(16)             :: w !< Buffer.

   write(w, '(I0)') i
   s = trim(w)
   endfunction itoa

   subroutine fail(what)
   !< Report a failed check.
   character(*), intent(in) :: what !< What failed.

   failures = failures + 1
   print '(A,I0,A)', '  FAIL rank ', wrank, ': '//what
   endsubroutine fail
endprogram elisha_test_solver_mpi
