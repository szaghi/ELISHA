!< ELISHA, the facade: one elliptic solver that chooses the spectral or the hierarchical path for the mesh.
module elisha_solver
!< ELISHA, the facade: one elliptic solver that chooses the spectral or the hierarchical path for the mesh.
!<
!< **Provenance: ADAM's `adam_flail_object` (ADAM 669d0bdd), its phase E0 facade.** `setup` chooses a solver for the
!< current mesh and `solve_packed` forwards to it with ADAM's call sequence, so a client gets bitwise the results of
!< the solvers called directly (`elisha_test_solver_mpi` holds it so on both paths). What changed (audit A4, A5):
!< - the solvers are private components; the path is the argument `path` (`ELISHA_PATH_AUTO`, `_SPECTRAL`,
!<   `_HIERARCHICAL`: ADAM's `FLAIL_BACKEND_AUTO`, `_FFT`, `_MULTIGRID`), and a backend (the device solvers of
!<   `elisha_device`) is chosen by the optional molds `spectral_mold` and `hierarchical_mold` of `setup`, where ADAM's
!<   client allocated the public components `fft` and `mg_mold` before calling it;
!< - what clients read from those components (PRISM: `fft%pcg`, `fft%iterations`, `fft%residual`; FLUME:
!<   `is_multigrid`, `iterations`) is the `elisha_diagnostics_object` returned by `diagnostics()`, and `is_hierarchical`;
!<   the boundary-face list that the values of `solve_packed(..., ghost)` refer to (ADAM's `mg%nbnd`, `mg%bnd_kind`,
!<   `mg%bnd_dir`, `mg%bnd_x`) is returned by `boundary_faces`;
!< - `solve(field, q, iv, ...)` is not ported: the source is packed (block-major, `i` fastest), the caller's layout is
!<   the adapter's to pack;
!< - every fatal refusal is a status (`stat`, `errmsg`): a spectral path on a mesh at more than one level
!<   (`ELISHA_ERR_MESH_LEVELS`), isolated boundaries with `bc` (`ELISHA_ERR_ISOLATED_BC`), the wide D(G) on the
!<   hierarchical path (`ELISHA_ERR_DG_AMR`), boundary values on the spectral path (`ELISHA_ERR_GHOST_VALUES`), a solve
!<   before the setup (`ELISHA_ERR_NOT_INITIALIZED`), an unknown path (`ELISHA_ERR_ARGUMENT`); the solvers' own
!<   refusals come up unchanged. ADAM's `option_name` (the client's input key in the fatal message) is gone: the caller
!<   words its own message from the status.
!<
!< **Paths.** `ELISHA_PATH_AUTO` takes the spectral solver when every leaf is at one level over the ranks
!< (`mesh%lmin == mesh%lmax`, after the collective `mesh%validate`), the hierarchical one otherwise;
!< `ELISHA_PATH_SPECTRAL` needs one level; `ELISHA_PATH_HIERARCHICAL` takes any 2:1 mesh (on a uniform mesh one V-cycle
!< is the exact spectral solve, and it accepts boundary values). The hierarchical solver's root is a spectral solver of
!< the dynamic type of `spectral_mold` when it is given (otherwise the hierarchical backend's default: the host one, or
!< cuFFT for the device backend). `setup` is called on the initial mesh and after every regrid: it destroys the solver in
!< use and builds the chosen one anew (so a hierarchical solve does not warm-start across setups).
!<
!< **Energy.** `solve_packed(..., energy)` returns `sum_cells f phi dV` on both paths (every rank): the hierarchical
!< solver's energy carries the cell volumes, the spectral solver's is `sum_cells f phi` and the facade multiplies it by
!< the one cell volume of the uniform mesh, as ADAM's facade did (`W = energy / 2` is the gravitational energy of
!< self-gravity).
use penf, only : I4P
use elisha_kind_parameters, only : R_P
use elisha_comm, only : elisha_comm_object
use elisha_mesh, only : elisha_mesh_object
use elisha_parameters, only : ELISHA_BC_PERIODIC, ELISHA_PATH_AUTO, ELISHA_PATH_SPECTRAL, ELISHA_PATH_HIERARCHICAL
use elisha_status, only : ELISHA_OK, ELISHA_ERR_ARGUMENT, ELISHA_ERR_ISOLATED_BC, ELISHA_ERR_MESH_LEVELS, &
                          ELISHA_ERR_GHOST_VALUES, ELISHA_ERR_NOT_INITIALIZED, elisha_set_error
use elisha_setup_checks, only : elisha_check_dg_order, elisha_check_path
use elisha_spectral_solver, only : elisha_spectral_solver_object
use elisha_hierarchical_solver, only : elisha_hierarchical_solver_object
implicit none
private
public :: elisha_diagnostics_object
public :: elisha_solver_object

type :: elisha_diagnostics_object
   !< What the last setup and solve of an `elisha_solver_object` report (a copy: plain data, every field public).
   !< Iteration counts and residuals are this rank's: the spectral CG runs on rank 0 (the other ranks report 0).
   integer(I4P) :: path=0_I4P          !< The path in use: ELISHA_PATH_SPECTRAL or _HIERARCHICAL; 0 before setup.
   integer(I4P) :: iterations=0_I4P    !< Iterations of the last solve: the FCG ones (hierarchical; with isolated
                                       !< boundaries the second solve of the James step), the CG ones (spectral, CG
                                       !< path), 0 for a direct spectral solve. ADAM's `iterations()` and `fft%iterations`.
   integer(I4P) :: iterations0=0_I4P   !< Hierarchical, isolated: FCG iterations of the zero-ghost solve of the James
                                       !< step (the first of the two); 0 otherwise.
   real(R_P)    :: residual=0._R_P     !< Relative max-norm residual of the last iterative solve (FCG or CG); 0 for a
                                       !< direct spectral solve. ADAM's `mg%residual` and `fft%residual`.
   real(R_P)    :: tolerance_used=0._R_P !< The stop tolerance of that solve: the hierarchical one's (the round-off
                                         !< floor of the source when above the requested tolerance), the spectral CG's;
                                         !< 0 for a direct spectral solve.
   logical      :: pcg=.false.         !< The spectral solve is the preconditioned CG of the wide D(G) with zero- or
                                       !< free-ghost faces (ADAM's `fft%pcg`); .false. on the hierarchical path.
   real(R_P)    :: wall(3)=0._R_P      !< Wall time (s) of the spectral `solve_blocks` on this rank, accumulated since the
                                       !< setup or `reset_wall`: gather (redistribution), solve, scatter (return). On the
                                       !< hierarchical path, its root solver's.
endtype elisha_diagnostics_object

type :: elisha_solver_object
   !< The elliptic solver: the spectral or the hierarchical solver, chosen for the mesh (see the module).
   private
   integer(I4P) :: path_in_use=0_I4P !< ELISHA_PATH_SPECTRAL or _HIERARCHICAL; 0: not set up.
   class(elisha_spectral_solver_object),     allocatable :: spectral     !< The spectral solver, on its path.
   class(elisha_hierarchical_solver_object), allocatable :: hierarchical !< The hierarchical solver, on its path.
   contains
      ! public methods
      procedure, pass(self) :: boundary_faces  !< The boundary faces of this rank (hierarchical path).
      procedure, pass(self) :: destroy         !< Destroy the solver in use.
      procedure, pass(self) :: diagnostics     !< The diagnostics of the last setup and solve.
      procedure, pass(self) :: is_hierarchical !< True when the setup chose the hierarchical solver.
      procedure, pass(self) :: path => solver_path !< The path in use (0 before setup).
      procedure, pass(self) :: reset_wall      !< Zero the accumulated wall times of the spectral solves.
      procedure, pass(self) :: setup           !< Choose and set up the solver for the current mesh.
      procedure, pass(self) :: solve_packed    !< Solve for phi from the packed interior source.
endtype elisha_solver_object

contains
   ! public methods
   subroutine boundary_faces(self, kinds, dir, x, stat, errmsg)
   !< The boundary faces of this rank's leaves, in the order `solve_packed(..., ghost)` takes their values (ADAM's
   !< `mg%nbnd`, `mg%bnd_kind`, `mg%bnd_dir`, `mg%bnd_x`): the kind of each (`ELISHA_BC_*`), its outward axis signed
   !< (-3..3) and where its value lives (the face centre, or the first ghost centre for ghost-Dirichlet). Hierarchical
   !< path only: the spectral one solves the homogeneous problem (`ELISHA_ERR_GHOST_VALUES`). Not collective.
   class(elisha_solver_object), intent(in)            :: self      !< The solver.
   integer(I4P), allocatable,   intent(out)           :: kinds(:)  !< Kind of each face [nbnd].
   integer(I4P), allocatable,   intent(out)           :: dir(:)    !< Outward axis of each face, signed [nbnd].
   real(R_P),    allocatable,   intent(out)           :: x(:,:)    !< Where each face's value lives [3,nbnd].
   integer(I4P),                intent(out)           :: stat      !< Status.
   character(:), allocatable,   intent(out), optional :: errmsg    !< Message.
   integer(I4P)                                       :: n         !< Faces.

   select case(self%path_in_use)
   case(ELISHA_PATH_HIERARCHICAL)
      n = self%hierarchical%nbnd
      allocate(kinds(n), dir(n), x(3,n))
      if (n > 0) then
         kinds = self%hierarchical%bnd_kind(1:n)
         dir   = self%hierarchical%bnd_dir(1:n)
         x     = self%hierarchical%bnd_x(:,1:n)
      endif
      stat = ELISHA_OK
   case(ELISHA_PATH_SPECTRAL)
      call elisha_set_error(stat, errmsg, ELISHA_ERR_GHOST_VALUES, 'elisha_solver_object%boundary_faces: the '// &
                            'spectral path solves the homogeneous problem (set up with ELISHA_PATH_HIERARCHICAL)')
   case default
      call elisha_set_error(stat, errmsg, ELISHA_ERR_NOT_INITIALIZED, 'elisha_solver_object%boundary_faces: '// &
                            'called before setup')
   endselect
   endsubroutine boundary_faces

   subroutine destroy(self)
   !< Destroy the solver in use (the caller still owns the communicator and the mesh).
   class(elisha_solver_object), intent(inout) :: self !< The solver.

   if (allocated(self%spectral)) then
      call self%spectral%destroy
      deallocate(self%spectral)
   endif
   if (allocated(self%hierarchical)) then
      call self%hierarchical%destroy
      deallocate(self%hierarchical)
   endif
   self%path_in_use = 0_I4P
   endsubroutine destroy

   pure function diagnostics(self) result(d)
   !< The diagnostics of the last setup and solve (see `elisha_diagnostics_object`); every field 0 before setup.
   class(elisha_solver_object), intent(in) :: self !< The solver.
   type(elisha_diagnostics_object)         :: d    !< Diagnostics.

   d%path = self%path_in_use
   select case(self%path_in_use)
   case(ELISHA_PATH_SPECTRAL)
      d%pcg = self%spectral%pcg
      if (d%pcg) then
         d%iterations     = self%spectral%iterations
         d%residual       = self%spectral%residual
         d%tolerance_used = self%spectral%tolerance
      endif
      d%wall = self%spectral%wall
   case(ELISHA_PATH_HIERARCHICAL)
      d%iterations     = self%hierarchical%iterations
      d%iterations0    = self%hierarchical%iterations0
      d%residual       = self%hierarchical%residual
      d%tolerance_used = self%hierarchical%tolerance_used
      if (allocated(self%hierarchical%fft)) d%wall = self%hierarchical%fft%wall
   endselect
   endfunction diagnostics

   pure function is_hierarchical(self)
   !< True when the setup chose the hierarchical solver (ADAM's `is_multigrid`).
   class(elisha_solver_object), intent(in) :: self            !< The solver.
   logical                                 :: is_hierarchical !< The hierarchical solver is in use.

   is_hierarchical = self%path_in_use == ELISHA_PATH_HIERARCHICAL
   endfunction is_hierarchical

   pure function solver_path(self) result(path)
   !< The path in use: ELISHA_PATH_SPECTRAL or _HIERARCHICAL; 0 before setup (or after a failed one).
   class(elisha_solver_object), intent(in) :: self !< The solver.
   integer(I4P)                            :: path !< Path.

   path = self%path_in_use
   endfunction solver_path

   subroutine reset_wall(self)
   !< Zero the accumulated wall times of the spectral solves (`diagnostics()%wall`).
   class(elisha_solver_object), intent(inout) :: self !< The solver.

   select case(self%path_in_use)
   case(ELISHA_PATH_SPECTRAL)
      self%spectral%wall = 0._R_P
   case(ELISHA_PATH_HIERARCHICAL)
      if (allocated(self%hierarchical%fft)) self%hierarchical%fft%wall = 0._R_P
   endselect
   endsubroutine reset_wall

   subroutine setup(self, mesh, comm, stat, errmsg, path, isolated, bc, dg_order, tolerance, max_iterations, root_cells, &
                    spectral_mold, hierarchical_mold)
   !< Choose and set up the solver for the current mesh: call it on the initial mesh and after every regrid (it rebuilds
   !< the solver). Collective over `comm`, which the solver keeps (a copy: the caller frees the communicator after
   !< `destroy`). The mesh is validated here (collectively) unless the caller did it.
   !<
   !< `path`: ELISHA_PATH_AUTO (default: spectral when every leaf is at one level, hierarchical otherwise),
   !< ELISHA_PATH_SPECTRAL (one level, else `ELISHA_ERR_MESH_LEVELS`), ELISHA_PATH_HIERARCHICAL. `isolated`, `bc`: the
   !< boundaries (free space; or the kind of every face, `elisha_parameters`), forwarded to the solver; isolated
   !< boundaries take no `bc` (`ELISHA_ERR_ISOLATED_BC`). `dg_order`: 0 the compact Laplacian (default), 2, 4, 6, 8 the
   !< wide D(G), spectral path only (`ELISHA_ERR_DG_AMR` on the hierarchical one). `tolerance`, `max_iterations`,
   !< `root_cells`: the hierarchical solver's controls (ignored on the spectral path). `spectral_mold`: the spectral
   !< solver's type (and the hierarchical root's); `hierarchical_mold`: the hierarchical solver's; without them, the host
   !< solvers. On failure the facade is left not set up (`path()` 0).
   class(elisha_solver_object),                     intent(inout)         :: self      !< The solver.
   type(elisha_mesh_object),                        intent(inout)         :: mesh      !< The mesh (validated here if
                                                                                       !< not yet).
   type(elisha_comm_object),                        intent(in)            :: comm      !< Communication context.
   integer(I4P),                                    intent(out)           :: stat      !< Status.
   character(:), allocatable,                       intent(out), optional :: errmsg    !< Message.
   integer(I4P),                                    intent(in),  optional :: path      !< ELISHA_PATH_AUTO (default),
                                                                                       !< _SPECTRAL, _HIERARCHICAL.
   logical,                                         intent(in),  optional :: isolated  !< Free-space boundaries.
   integer(I4P),                                    intent(in),  optional :: bc(2,3)   !< Kinds per face [lo/hi, axis].
   integer(I4P),                                    intent(in),  optional :: dg_order  !< 0 compact; 2, 4, 6, 8: D(G).
   real(R_P),                                       intent(in),  optional :: tolerance !< Hierarchical: relative
                                                                                       !< residual to stop.
   integer(I4P),                                    intent(in),  optional :: max_iterations !< Hierarchical: FCG
                                                                                            !< iterations at most.
   integer(I4P),                                    intent(in),  optional :: root_cells !< Hierarchical: most cells of
                                                                                        !< the root.
   class(elisha_spectral_solver_object),            intent(in),  optional :: spectral_mold !< The spectral backend.
   class(elisha_hierarchical_solver_object),        intent(in),  optional :: hierarchical_mold !< The hierarchical
                                                                                               !< backend.
   integer(I4P)                                                           :: path_     !< The path asked.
   integer(I4P)                                                           :: dg_order_ !< The operator's order.
   integer(I4P)                                                           :: bc_(2,3)  !< The kinds (periodic if none).
   logical                                                                :: isolated_ !< Free-space boundaries.
   logical                                                                :: uniform   !< Every leaf at one level.
   character(64)                                                          :: w         !< Message buffer.
   character(*), parameter :: HERE='elisha_solver_object%setup' !< For the messages.

   call self%destroy
   if (.not.comm%is_initialized) then
      call elisha_set_error(stat, errmsg, ELISHA_ERR_NOT_INITIALIZED, HERE//': the communication context is not '// &
                            'initialised')
      return
   endif
   ! the argument refusals (the same on every rank: checks of the same arguments)
   path_ = ELISHA_PATH_AUTO ; if (present(path)) path_ = path
   if (all(path_ /= [ELISHA_PATH_AUTO, ELISHA_PATH_SPECTRAL, ELISHA_PATH_HIERARCHICAL])) then
      write(w, '(A,I0)') 'path ', path_
      call elisha_set_error(stat, errmsg, ELISHA_ERR_ARGUMENT, HERE//': '//trim(w)//' (ELISHA_PATH_AUTO, _SPECTRAL '// &
                            'or _HIERARCHICAL)')
      return
   endif
   isolated_ = .false. ; if (present(isolated)) isolated_ = isolated
   if (isolated_ .and. present(bc)) then
      call elisha_set_error(stat, errmsg, ELISHA_ERR_ISOLATED_BC, HERE//': isolated boundaries take no bc')
      return
   endif
   dg_order_ = 0_I4P ; if (present(dg_order)) dg_order_ = dg_order
   call elisha_check_dg_order(dg_order_, stat, errmsg) ; if (stat /= ELISHA_OK) return
   bc_ = ELISHA_BC_PERIODIC ; if (present(bc)) bc_ = bc
   ! the levels over the ranks (ADAM's reduction): the mesh's validation, collective
   if (.not.mesh%is_validated) then
      call mesh%validate(comm, stat, errmsg)
      if (stat /= ELISHA_OK) return
   endif
   uniform = mesh%lmin == mesh%lmax
   if (path_ == ELISHA_PATH_SPECTRAL .and. .not.uniform) then
      write(w, '(A,I0,A,I0)') 'the leaves sit at levels ', mesh%lmin, ' to ', mesh%lmax
      call elisha_set_error(stat, errmsg, ELISHA_ERR_MESH_LEVELS, HERE//': path spectral needs a uniform mesh ('// &
                            trim(w)//'): use ELISHA_PATH_AUTO or ELISHA_PATH_HIERARCHICAL')
      return
   endif
   if (path_ == ELISHA_PATH_SPECTRAL .or. (path_ == ELISHA_PATH_AUTO .and. uniform)) then
      if (present(spectral_mold)) then
         allocate(self%spectral, mold=spectral_mold)
      else
         allocate(elisha_spectral_solver_object :: self%spectral)
      endif
      call self%spectral%setup(mesh=mesh, comm=comm, stat=stat, errmsg=errmsg, isolated=isolated, bc=bc, &
                               dg_order=dg_order)
      if (stat /= ELISHA_OK) then
         call self%destroy
         return
      endif
      self%path_in_use = ELISHA_PATH_SPECTRAL
   else
      call elisha_check_path(ELISHA_PATH_HIERARCHICAL, uniform, dg_order_, bc_, isolated_, .false., .false., .false., &
                             stat, errmsg)
      if (stat /= ELISHA_OK) return
      if (present(hierarchical_mold)) then
         allocate(self%hierarchical, mold=hierarchical_mold)
      else
         allocate(elisha_hierarchical_solver_object :: self%hierarchical)
      endif
      call self%hierarchical%setup(mesh=mesh, comm=comm, stat=stat, errmsg=errmsg, tolerance=tolerance,         &
                                   max_iterations=max_iterations, coarse=spectral_mold, isolated=isolated, bc=bc, &
                                   root_cells=root_cells)
      if (stat /= ELISHA_OK) then
         call self%destroy
         return
      endif
      self%path_in_use = ELISHA_PATH_HIERARCHICAL
   endif
   endsubroutine setup

   subroutine solve_packed(self, mesh, buf, factor, phi, stat, errmsg, energy, ghost, on_device)
   !< Solve for phi from the packed interior source of this rank's blocks (block-major, `i` fastest), every ghost layer
   !< of phi filled; optionally the energy `sum_cells f phi dV` (every rank). `ghost`: the value of each of this rank's
   !< boundary faces (`boundary_faces`, of its kind), hierarchical path only (`ELISHA_ERR_GHOST_VALUES` on the spectral
   !< one, which solves the homogeneous problem: an argument every rank passes alike, refused before any collective).
   !< `on_device` (M7, audit C4): `buf` and `phi` are present on the device (OpenACC; `phi` contiguous), forwarded to the
   !< solver in use: a device backend (the molds of `setup`) stages them around its solve, bitwise the result from host
   !< arrays; a host solver returns `ELISHA_ERR_ARGUMENT`. Collective.
   class(elisha_solver_object), intent(inout)         :: self    !< The solver.
   type(elisha_mesh_object),    intent(in)            :: mesh    !< The mesh (this rank's blocks, the setup's).
   real(R_P),                   intent(in)            :: buf(1:) !< Packed interior source.
   real(R_P),                   intent(in)            :: factor  !< Source factor (4 pi G for gravity).
   real(R_P),    intent(inout) :: phi(1-mesh%ngc:,1-mesh%ngc:,1-mesh%ngc:,1:) !< Potential [i,j,k,nb], ghosts included.
   integer(I4P),                intent(out)           :: stat    !< Status.
   character(:), allocatable,   intent(out), optional :: errmsg  !< Message.
   real(R_P),                   intent(out), optional :: energy  !< sum_cells f phi dV (all ranks).
   real(R_P),                   intent(in),  optional :: ghost(1:) !< Boundary value per boundary face (hierarchical).
   logical,                     intent(in),  optional :: on_device !< buf and phi on the device (default: host).

   select case(self%path_in_use)
   case(ELISHA_PATH_HIERARCHICAL)
      call self%hierarchical%solve_packed(mesh=mesh, buf=buf, factor=factor, phi=phi, stat=stat, errmsg=errmsg, &
                                          energy=energy, ghost=ghost, on_device=on_device)
   case(ELISHA_PATH_SPECTRAL)
      if (present(ghost)) then
         call elisha_set_error(stat, errmsg, ELISHA_ERR_GHOST_VALUES, 'elisha_solver_object%solve_packed: boundary '// &
                               'values need the hierarchical path; set it up with ELISHA_PATH_HIERARCHICAL (on a '// &
                               'uniform mesh one V-cycle is the exact spectral solve)')
         return
      endif
      call self%spectral%solve_packed(mesh=mesh, buf=buf, factor=factor, phi=phi, stat=stat, errmsg=errmsg, &
                                      energy=energy, on_device=on_device)
      if (stat /= ELISHA_OK) return
      if (present(energy)) energy = energy * product(self%spectral%dxyz) ! uniform mesh: one cell volume
   case default
      call elisha_set_error(stat, errmsg, ELISHA_ERR_NOT_INITIALIZED, 'elisha_solver_object%solve_packed: called '// &
                            'before setup')
   endselect
   endsubroutine solve_packed
endmodule elisha_solver
