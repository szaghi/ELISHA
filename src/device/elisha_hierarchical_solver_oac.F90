!< ELISHA, the hierarchical solver's device backend: OpenACC on NVIDIA GPUs, the root on cuFFT.
module elisha_hierarchical_solver_oac
!< ELISHA, the hierarchical solver's device backend: OpenACC on NVIDIA GPUs, the root on cuFFT.
!<
!< **Provenance: ADAM's `adam_fnl_poisson_mg_object` (ADAM 669d0bdd), plumbing changed, arithmetic not.** It extends
!< `elisha_hierarchical_solver_object` (ADAM's `poisson_mg_object` ported) as ADAM's extends its host object, and
!< overrides the same hooks. The device code is ADAM's: every OpenACC construct and clause, every kernel's loops, their
!< order and their reductions, the placement of the operators and vectors (`enter`/`exit data` in the same order), the
!< halo buffers sized for the largest operator and the GPU-aware `MPI_ALLTOALLV` on them, so the potential, the
!< iterations and the residuals are meant to be ADAM's device ones bitwise, on the same GPU and compiler
!< (`src/tests/device/elisha_test_device_parity_mpi`, against the verbatim ADAM twin). What changed:
!< - `initialize(grid, field, ...)` is `setup(mesh, comm, stat, errmsg, ...)`, the coarsest level (and the James step's
!<   Green's function) on `elisha_spectral_solver_cufft_object` without `coarse` (ADAM: `poisson_fft_fnl_object`); as
!<   ADAM's, it releases the device copies of a previous setup first (`destroy` through the dynamic type), sets up
!<   through the parent component (whose `work_place` is the host's no-op) and places them after;
!< - ELISHA's hook interfaces: `apply`, `smooth`, `work_place` return `stat`; `vec_dot` and `vec_relmax` pass `self`
!<   (the communicator) and return `stat`; their reductions are `elisha_comm_object%sum_real8` and `max_real8`, ADAM's
!<   `MPI_ALLREDUCE` in place of one `MPI_REAL8` (sum, max), on the solver's communicator instead of `MPI_COMM_WORLD`;
!< - the halo exchange is the operator's `elisha_halo_plan%exchange_r` (ADAM: `MPI_ALLTOALLV`, `MPI_REAL8`,
!<   `MPI_COMM_WORLD`; since M7, audit B1, `MPI_Neighbor_alltoallv` on the plan's graph communicator, the same buffers,
!<   displacements and values), called inside the same `host_data use_device(sb, hb)` region with the same buffers:
!<   its dummies are assumed-size, so the device addresses pass untouched to MPI; ADAM ignored the MPI error code, here
!<   it is `stat`;
!< - `work_place` before `setup` returns `ELISHA_ERR_NOT_INITIALIZED` (ADAM stopped), as the host hook does;
!< - ELISHA's host changes stay host-side and change no device arithmetic: the host's persistent halo buffers
!<   (`xsend`, `xhalo`, audit C1) are allocated and unused here (the device ones, `dsb`, `dhb`, were ADAM's already);
!<   the red-black colour assertion is the host assembly's, which this backend inherits (the parallel colour sweep
!<   relies on it);
!< - one M7 change (audit C4): the V-cycle's root stays on the device when the root solver can (`root_solve`: the
!<   cuFFT solver on its redundant path, `has_device_redundant`): `e = L^-1 (r / V)` with the division a device kernel
!<   (the same IEEE division, `r / v0`, per cell) into the level temporary `lt1(lbot)`, free at that point of the
!<   V-cycle, and `solve_redundant_device` writing `le(lbot)` on the device; ADAM (and the host fallback, any other
!<   root) went through the host at every V-cycle. And `solve_packed(on_device)` (`stage_source`,
!<   `stage_potential`): the device source copied to a host copy, `phi` to the host and back, around the host-side
!<   right-hand side and ghost fill, which stay on the host (their sums and products in the host's order);
!< - one M7 change (audit C2), this backend's default: `level_operator=ELISHA_OPERATOR_STENCIL` (the host's default
!<   is CSR), the level operators' block-interior rows applied and smoothed as the host's stencil (`spmv_sten_dev`,
!<   `sweep_sten_dev`: one row per thread, the cell decoded from the row, the same terms in the same order as its CSR
!<   entries), the other rows by CSR in the same kernel: bitwise ADAM FNL's results on the A30, 10-15% faster
!<   (medooza job 1369037); `level_operator=ELISHA_OPERATOR_CSR` selects ADAM's loops;
!< - one M7 change (audit C3): the colour is the host's integer (`ivec`, 0 or 1), placed by `place_i` and compared as
!<   such in `sweep_dev`; ADAM's was a real compared through `nint`. It selects rows only, so no result changes, and
!<   the kernel reads 4 bytes per row for it instead of 8;
!< - the host's symmetric V-cycle (ELISHA notes E1c, the default, `vcycle=ELISHA_VCYCLE_ADAM` for ADAM FNL's): its
!<   extra operator `etl` (E_l^T) goes to the device with the others and is applied by `apply_oac` (CSR).
!< The `R8P` of ADAM is ELISHA's `R_P` (R8P).
!<
!< ADAM's description, unchanged: the leaves, the hierarchy and every operator are assembled on the host; `work_place`
!< then puts the operators (the composite operator, the ghost fill, the level operators, ghost terms and transfers,
!< CSR), the level diagonals and colours, the cell volumes, the maps of the leaves into the level spaces and the work
!< vectors on the device, where they stay until `destroy`. The FCG iteration and the V-cycle run there through the
!< hooks: `apply` (the halo values packed on the device, exchanged on the device buffers, none on one rank, then one
!< thread per row, its entries summed in the host's order); `smooth` (the red-black Gauss-Seidel sweeps, one kernel per
!< colour after its halo exchange: a row couples only to cells of the other colour, so the parallel update of a colour
!< is the host's sequential one); the vector hooks (fill, copy, combinations, dots and the max-norm by device
!< reductions, the permutations of the leaves into the levels). The root goes through the host to the spectral solver,
!< and so do the right-hand side, the James step and the ghost fill's placement. The results equal the host's to
!< round-off: the dots sum in another order and the device contracts products into FMAs.
use penf, only : I4P
use elisha_kind_parameters, only : R_P
use elisha_comm, only : elisha_comm_object, elisha_halo_plan
use elisha_mesh, only : elisha_mesh_object
use elisha_spectral_solver, only : elisha_spectral_solver_object
use elisha_spectral_solver_cufft, only : elisha_spectral_solver_cufft_object
use elisha_hierarchical_solver, only : elisha_hierarchical_solver_object, dcsr, hierarchical_solve_fcg, &
                                      hierarchical_solve_fcg_device
use elisha_parameters, only : ELISHA_OPERATOR_STENCIL, ELISHA_ASSEMBLY_HOST, ELISHA_ASSEMBLY_DEVICE, &
                              ELISHA_BC_DIRICHLET_GHOST
use elisha_status, only : ELISHA_OK, ELISHA_ERR_NOT_INITIALIZED, ELISHA_ERR_ARGUMENT, elisha_set_error
implicit none
private
public :: elisha_hierarchical_solver_oac_object

type, extends(elisha_hierarchical_solver_object) :: elisha_hierarchical_solver_oac_object
   !< The hierarchical solver on OpenACC (see the module).
   logical                :: on_device=.false. !< The operators and the work vectors are on the device.
   real(R_P), allocatable :: dsb(:)            !< Device: the halo send buffer (the largest operator's).
   real(R_P), allocatable :: dhb(:)            !< Device: the halo receive buffer (the largest operator's).
   ! the device assembly (ELISHA notes "device assembly")
   real(R_P),    allocatable :: dbuf(:)          !< Device: a host-array solve's source, uploaded.
   real(R_P),    allocatable :: dghost(:)        !< Device: a solve's boundary values, uploaded.
   integer(I4P), allocatable :: brow(:)          !< The rows with boundary faces [nbrow].
   integer(I4P), allocatable :: bptr(:)          !< Their faces' pointers into bidx [nbrow+1].
   integer(I4P), allocatable :: bidx(:)          !< The faces of each row, in face order (the host's order) [nbnd].
   integer(I4P)              :: nbrow=0          !< Rows with boundary faces.
   contains
      ! public methods
      procedure, pass(self) :: setup => setup_oac !< As the host's, the coarsest level on cuFFT by default.
      ! backend hooks
      procedure, pass(self) :: apply       => apply_oac       !< y = M x on the device.
      procedure, pass(self) :: smooth      => smooth_oac      !< Red-black Gauss-Seidel on the device.
      procedure, nopass     :: vec_fill    => vec_fill_oac    !< x(1:n) = a on the device.
      procedure, nopass     :: vec_copy    => vec_copy_oac    !< y(1:n) = x(1:n) on the device.
      procedure, nopass     :: vec_lincomb => vec_lincomb_oac !< w = x + a y on the device.
      procedure, nopass     :: vec_axpy    => vec_axpy_oac    !< y = y + a x on the device.
      procedure, pass(self) :: vec_dot     => vec_dot_oac     !< sum x y on the device and the ranks.
      procedure, pass(self) :: vec_relmax  => vec_relmax_oac  !< max |r / vol| on the device and the ranks.
      procedure, nopass     :: vec_permute => vec_permute_oac !< y(iy) = x(ix) on the device.
      procedure, nopass     :: vec_update  => vec_update_oac  !< A work vector to or from the device.
      procedure, pass(self) :: work_place  => work_place_oac  !< The operators and work vectors to or from the device.
      procedure, pass(self) :: root_solve  => root_solve_oac  !< The V-cycle's root on the device (M7, audit C4).
      procedure, pass(self) :: stage_source    => stage_source_oac    !< solve_packed(on_device): the source.
      procedure, pass(self) :: stage_potential => stage_potential_oac !< solve_packed(on_device): phi.
      procedure, pass(self) :: solve_fcg       => solve_fcg_oac       !< The FCG solve: the device assembly or ADAM's.
      procedure, pass(self) :: assemble_rhs    => assemble_rhs_oac    !< The right-hand side, on the device.
      procedure, pass(self) :: fill_potential  => fill_potential_oac  !< phi and the energy, on the device.
endtype elisha_hierarchical_solver_oac_object

contains
   ! public methods
   subroutine setup_oac(self, mesh, comm, stat, errmsg, tolerance, max_iterations, coarse, isolated, bc, root_cells, &
                        level_operator, vcycle, assembly)
   !< As `elisha_hierarchical_solver_object%setup`; without `coarse`, the coarsest level (and the James step's Green's
   !< function) on cuFFT; without `level_operator`, the stencil (`ELISHA_OPERATOR_STENCIL`, the host's default is CSR);
   !< without `assembly`, the device assembly (`ELISHA_ASSEMBLY_DEVICE`; `ELISHA_ASSEMBLY_HOST` is ADAM FNL's, bitwise).
   !< Then the operators and the work vectors go to the device.
   class(elisha_hierarchical_solver_oac_object), intent(inout)        :: self      !< The solver.
   type(elisha_mesh_object),       intent(inout)                      :: mesh      !< The mesh (validated if not yet).
   type(elisha_comm_object),       intent(in)                         :: comm      !< Communication context.
   integer(I4P),                   intent(out)                        :: stat      !< Status.
   character(:), allocatable,      intent(out),          optional     :: errmsg    !< Message.
   real(R_P),                      intent(in),           optional     :: tolerance !< Relative residual to stop.
   integer(I4P),                   intent(in),           optional     :: max_iterations !< FCG iterations at most.
   class(elisha_spectral_solver_object), intent(in),     optional     :: coarse    !< Mold of the root solver.
   logical,                        intent(in),           optional     :: isolated  !< Free-space boundaries (James).
   integer(I4P),                   intent(in),           optional     :: bc(2,3)   !< Kinds per face [lo/hi, axis].
   integer(I4P),                   intent(in),           optional     :: root_cells !< Most cells of the root.
   integer(I4P),                   intent(in),           optional     :: level_operator !< The level operators' form
                                                                                         !< (default: the stencil).
   integer(I4P),                   intent(in),           optional     :: vcycle    !< The V-cycle (default: symmetric).
   integer(I4P),                   intent(in),           optional     :: assembly  !< Where a solve is assembled.
   type(elisha_spectral_solver_cufft_object)                          :: cufft_mold !< The cuFFT root, by default.
   integer(I4P)                                                       :: code      !< Status of the placement.
   integer(I4P)                                                       :: lop       !< The level operators' form.
   integer(I4P)                                                       :: asm       !< Where a solve is assembled.

   ! through the parent component the passed object is an elisha_hierarchical_solver_object: its destroy and
   ! work_place are the host's (no-ops), so the device copies of a previous setup are released here first and the new
   ! ones placed after
   call self%destroy
   lop = ELISHA_OPERATOR_STENCIL ; if (present(level_operator)) lop = level_operator
   asm = ELISHA_ASSEMBLY_DEVICE ; if (present(assembly)) asm = assembly
   if (all(asm /= [ELISHA_ASSEMBLY_HOST, ELISHA_ASSEMBLY_DEVICE])) then ! every rank alike: the same argument
      call elisha_set_error(stat, errmsg, ELISHA_ERR_ARGUMENT, 'elisha_hierarchical_solver_oac_object%setup: '// &
                            'assembly is not ELISHA_ASSEMBLY_HOST or ELISHA_ASSEMBLY_DEVICE')
      return
   endif
   if (present(coarse)) then
      call self%elisha_hierarchical_solver_object%setup(mesh=mesh, comm=comm, stat=stat, errmsg=errmsg,           &
                                                        tolerance=tolerance, max_iterations=max_iterations,      &
                                                        coarse=coarse, isolated=isolated, bc=bc, root_cells=root_cells, &
                                                        level_operator=lop, vcycle=vcycle)
   else
      call self%elisha_hierarchical_solver_object%setup(mesh=mesh, comm=comm, stat=stat, errmsg=errmsg,           &
                                                        tolerance=tolerance, max_iterations=max_iterations,      &
                                                        coarse=cufft_mold, isolated=isolated, bc=bc,             &
                                                        root_cells=root_cells, level_operator=lop, vcycle=vcycle)
   endif
   if (stat /= ELISHA_OK) return
   call self%work_place(attach=.true., stat=code)
   if (code /= ELISHA_OK) then
      call elisha_set_error(stat, errmsg, code, 'elisha_hierarchical_solver_oac_object%setup: work_place failed')
      call self%destroy
      return
   endif
   self%assembly = asm ! after the host setup, which refuses the device assembly (it has no device)
   endsubroutine setup_oac

   ! backend hooks
   subroutine apply_oac(self, m, x, y, stat)
   !< y = M x on the device: the halo of x packed, exchanged (GPU-aware), then the local rows.
   class(elisha_hierarchical_solver_oac_object), intent(inout) :: self  !< The solver.
   type(dcsr),                                   intent(in)    :: m     !< Operator (on the device).
   real(R_P),                                    intent(in)    :: x(1:) !< Local column-space vector (on the device).
   real(R_P),                                    intent(inout) :: y(1:) !< Result (rows, on the device).
   integer(I4P),                                 intent(out)   :: stat  !< Status.

   call halo_dev(self, m, x, stat) ; if (stat /= ELISHA_OK) return
   if (m%nrow == 0) return
   if (m%stencil) then ! the block-interior rows as the stencil (M7, audit C2)
      call spmv_sten_dev(nrow=m%nrow, nloc=m%nloc, nnz=size(m%col), nx=size(x), ny=size(y), nh=size(self%dhb), &
                         rowptr=m%rowptr, col=m%col, val=m%val, x=x, h=self%dhb, y=y, nl=m%snl, lo=m%slo,     &
                         hi=m%shi, nd=m%snd, off=m%soff, w=m%sw)
   else
      call spmv_dev(nrow=m%nrow, nloc=m%nloc, nnz=size(m%col), nx=size(x), ny=size(y), nh=size(self%dhb), &
                    rowptr=m%rowptr, col=m%col, val=m%val, x=x, h=self%dhb, y=y)
   endif
   endsubroutine apply_oac

   subroutine smooth_oac(self, lv, e, rhs, order, stat)
   !< Red-black Gauss-Seidel sweeps on level lv on the device: per colour, the halo of e, then every row of the colour
   !< (they read only the other colour's cells).
   class(elisha_hierarchical_solver_oac_object), intent(inout) :: self    !< The solver.
   integer(I4P),                                 intent(in)    :: lv      !< Level.
   real(R_P),                                    intent(inout) :: e(1:)   !< Correction (on the device).
   real(R_P),                                    intent(in)    :: rhs(1:) !< Right-hand side (on the device).
   integer(I4P),                                 intent(in)    :: order   !< 0: colour 0 first, 1: colour 1 first.
   integer(I4P),                                 intent(out)   :: stat    !< Status.
   integer(I4P)                                                :: sw, cc_ !< Counters.

   stat = ELISHA_OK
   do sw=1, self%sweeps
      do cc_=0, 1
         call halo_dev(self, self%al(lv), e, stat) ; if (stat /= ELISHA_OK) return
         if (self%al(lv)%nrow == 0) cycle
         if (self%al(lv)%stencil) then ! the block-interior rows as the stencil (M7, audit C2)
            associate(m=>self%al(lv))
            call sweep_sten_dev(nrow=m%nrow, nloc=m%nloc, nnz=size(m%col), ne=size(e), nr=size(rhs), nh=size(self%dhb), &
                                nd=size(self%diag(lv)%v), colr=mod(cc_ + order, 2), rowptr=m%rowptr, col=m%col,         &
                                val=m%val, diag=self%diag(lv)%v, colour=self%colour(lv)%v, rhs=rhs, h=self%dhb, e=e,    &
                                nl=m%snl, lo=m%slo, hi=m%shi, ns=m%snd, off=m%soff, w=m%sw)
            endassociate
         else
            call sweep_dev(nrow=self%al(lv)%nrow, nloc=self%al(lv)%nloc, nnz=size(self%al(lv)%col), ne=size(e),  &
                           nr=size(rhs), nh=size(self%dhb), nd=size(self%diag(lv)%v), colr=mod(cc_ + order, 2), &
                           rowptr=self%al(lv)%rowptr, col=self%al(lv)%col, val=self%al(lv)%val,                &
                           diag=self%diag(lv)%v, colour=self%colour(lv)%v, rhs=rhs, h=self%dhb, e=e)
         endif
      enddo
   enddo
   endsubroutine smooth_oac

   subroutine vec_fill_oac(x, a, n)
   !< x(1:n) = a on the device.
   real(R_P),    intent(inout) :: x(1:) !< Vector (on the device).
   real(R_P),    intent(in)    :: a     !< Value.
   integer(I4P), intent(in)    :: n     !< Entries.

   if (n > 0) call fill_dev(size(x), x, a, n)
   endsubroutine vec_fill_oac

   subroutine vec_copy_oac(x, y, n)
   !< y(1:n) = x(1:n) on the device.
   real(R_P),    intent(in)    :: x(1:) !< Source (on the device).
   real(R_P),    intent(inout) :: y(1:) !< Destination (on the device).
   integer(I4P), intent(in)    :: n     !< Entries.

   if (n > 0) call lincomb_dev(size(x), x, 0._R_P, size(x), x, size(y), y, n, .false.)
   endsubroutine vec_copy_oac

   subroutine vec_lincomb_oac(x, a, y, w, n)
   !< w(1:n) = x(1:n) + a y(1:n) on the device.
   real(R_P),    intent(in)    :: x(1:) !< First vector (on the device).
   real(R_P),    intent(in)    :: a     !< Coefficient.
   real(R_P),    intent(in)    :: y(1:) !< Second vector (on the device).
   real(R_P),    intent(inout) :: w(1:) !< Result (on the device).
   integer(I4P), intent(in)    :: n     !< Entries.

   if (n > 0) call lincomb_dev(size(x), x, a, size(y), y, size(w), w, n, .true.)
   endsubroutine vec_lincomb_oac

   subroutine vec_axpy_oac(a, x, y, n)
   !< y(1:n) = y(1:n) + a x(1:n) on the device.
   real(R_P),    intent(in)    :: a     !< Coefficient.
   real(R_P),    intent(in)    :: x(1:) !< Vector (on the device).
   real(R_P),    intent(inout) :: y(1:) !< Updated vector (on the device).
   integer(I4P), intent(in)    :: n     !< Entries.

   if (n > 0) call axpy_dev(a, size(x), x, size(y), y, n)
   endsubroutine vec_axpy_oac

   function vec_dot_oac(self, x, y, n, stat) result(d)
   !< sum_i x_i y_i over the local n entries (a device reduction) and the ranks.
   class(elisha_hierarchical_solver_oac_object), intent(in)  :: self  !< The solver (its communicator).
   real(R_P),                                    intent(in)  :: x(1:) !< First vector (on the device).
   real(R_P),                                    intent(in)  :: y(1:) !< Second vector (on the device).
   integer(I4P),                                 intent(in)  :: n     !< Local entries.
   integer(I4P),                                 intent(out) :: stat  !< Status.
   real(R_P)                                                 :: d     !< Result.
   real(R_P)                                                 :: x1(1) !< d, reduced in place.

   d = 0._R_P
   if (n > 0) d = dot_dev(size(x), x, size(y), y, n)
   x1 = d
   call self%comm%sum_real8(x1, stat)
   d = x1(1)
   endfunction vec_dot_oac

   function vec_relmax_oac(self, r, vol, n, fmax, stat) result(rm)
   !< max |r / vol| over the local n entries (a device reduction) and the ranks, over fmax (when positive).
   class(elisha_hierarchical_solver_oac_object), intent(in)  :: self    !< The solver (its communicator).
   real(R_P),                                    intent(in)  :: r(1:)   !< Residual (on the device).
   real(R_P),                                    intent(in)  :: vol(1:) !< Cell volumes (on the device).
   integer(I4P),                                 intent(in)  :: n       !< Local entries.
   real(R_P),                                    intent(in)  :: fmax    !< Normalisation.
   integer(I4P),                                 intent(out) :: stat    !< Status.
   real(R_P)                                                 :: rm      !< Result.
   real(R_P)                                                 :: x1(1)   !< rm, reduced in place.

   rm = 0._R_P
   if (n > 0) rm = relmax_dev(size(r), r, size(vol), vol, n)
   x1 = rm
   call self%comm%max_real8(x1, stat)
   rm = x1(1)
   if (fmax > 0._R_P) rm = rm / fmax
   endfunction vec_relmax_oac

   subroutine vec_permute_oac(x, ix, y, iy, n)
   !< y(iy(i)) = x(ix(i)), i = 1..n, on the device.
   real(R_P),    intent(in)    :: x(1:)  !< Source (on the device).
   integer(I4P), intent(in)    :: ix(1:) !< Source entries (on the device).
   real(R_P),    intent(inout) :: y(1:)  !< Destination (on the device).
   integer(I4P), intent(in)    :: iy(1:) !< Destination entries (on the device).
   integer(I4P), intent(in)    :: n      !< Entries.

   if (n > 0) call permute_dev(size(x), x, size(ix), ix, size(y), y, size(iy), iy, n)
   endsubroutine vec_permute_oac

   subroutine vec_update_oac(x, to_device)
   !< A work vector to (`to_device`) or from the device.
   real(R_P), intent(inout) :: x(1:)     !< Work vector (on the device).
   logical,   intent(in)    :: to_device !< Direction.

   call update_dev(size(x), x, to_device)
   endsubroutine vec_update_oac

   subroutine work_place_oac(self, attach, stat)
   !< Put (`attach`) the operators, the level diagonals, colours and maps, the volumes and the work vectors on the
   !< device, with the halo buffers sized for the largest operator; or release them.
   class(elisha_hierarchical_solver_oac_object), intent(inout) :: self   !< The solver.
   logical,                                      intent(in)    :: attach !< Put, or release.
   integer(I4P),                                 intent(out)   :: stat   !< Status.
   integer(I4P)                                                :: lv, ns, nh !< Level, buffer sizes.

   if (.not.self%is_initialized) then
      stat = ELISHA_ERR_NOT_INITIALIZED
      return
   endif
   stat = ELISHA_OK
   if (attach .eqv. self%on_device) return
   if (attach) then
      ns = max(size(self%a%sidx), size(self%g%sidx)) ; nh = max(self%a%nhalo, self%g%nhalo)
      do lv=self%lbot+1, self%lmax
         ns = max(ns, size(self%al(lv)%sidx), size(self%el(lv)%sidx), size(self%rl(lv)%sidx), size(self%pl(lv)%sidx), &
                  size(self%etl(lv)%sidx))
         nh = max(nh, self%al(lv)%nhalo, self%el(lv)%nhalo, self%rl(lv)%nhalo, self%pl(lv)%nhalo, self%etl(lv)%nhalo)
      enddo
      allocate(self%dsb(max(ns,1)), self%dhb(max(nh,1)))
      self%dsb = 0._R_P ; self%dhb = 0._R_P
      call place_r(self%dsb, .true.) ; call place_r(self%dhb, .true.)
   endif
   call place_csr(self%a, attach) ; call place_csr(self%g, attach)
   do lv=self%lbot+1, self%lmax
      call place_csr(self%al(lv), attach) ; call place_csr(self%el(lv), attach)
      call place_csr(self%rl(lv), attach) ; call place_csr(self%pl(lv), attach)
      call place_csr(self%etl(lv), attach)
      call place_r(self%diag(lv)%v, attach) ; call place_i(self%colour(lv)%v, attach)
   enddo
   do lv=self%lbot, self%lmax
      call place_r(self%lr(lv)%v, attach) ; call place_r(self%le(lv)%v, attach)
      call place_r(self%lt1(lv)%v, attach) ; call place_r(self%lt2(lv)%v, attach)
      call place_i(self%mlev(lv)%v, attach) ; call place_i(self%mleaf(lv)%v, attach)
   enddo
   call place_r(self%vol, attach) ; call place_r(self%u, attach) ; call place_r(self%u0, attach)
   call place_r(self%wf, attach) ; call place_r(self%wr, attach) ; call place_r(self%wz, attach)
   call place_r(self%wp, attach) ; call place_r(self%wq, attach) ; call place_r(self%wpp, attach)
   call place_r(self%wqp, attach) ; call place_r(self%w1, attach) ; call place_r(self%wg, attach)
   if (attach) call group_boundary_rows(self)
   call place_r(self%dbuf, attach) ; call place_r(self%dghost, attach)
   call place_i(self%brow, attach) ; call place_i(self%bptr, attach) ; call place_i(self%bidx, attach)
   if (allocated(self%bnd_w))  call place_r(self%bnd_w, attach)
   if (allocated(self%bnd_gs)) call place_r(self%bnd_gs, attach)
   if (allocated(self%bnd_kind)) call place_i(self%bnd_kind, attach)
   if (allocated(self%g_bnd))  call place_i(self%g_bnd, attach)
   if (allocated(self%g_dst))  call place_i2(self%g_dst, attach)
   if (.not.attach) then
      call place_r(self%dsb, .false.) ; call place_r(self%dhb, .false.)
      deallocate(self%dsb, self%dhb)
      deallocate(self%dbuf, self%dghost, self%brow, self%bptr, self%bidx)
      self%nbrow = 0
   endif
   self%on_device = attach
   endsubroutine work_place_oac

   subroutine root_solve_oac(self, n, nc, v0, stat, errmsg)
   !< The V-cycle's root on the device when the root solver can (the cuFFT solver on its redundant path): the source
   !< `lr(lbot) / v0` into `lt1(lbot)` by a device kernel, the solve into `le(lbot)`, on the device; otherwise the host
   !< path (`elisha_hierarchical_solver_object%root_solve`, ADAM's).
   class(elisha_hierarchical_solver_oac_object), intent(inout)         :: self   !< The solver.
   integer(I4P),                                 intent(in)            :: n      !< Root blocks of this rank.
   integer(I4P),                                 intent(in)            :: nc     !< Root cells of this rank.
   real(R_P),                                    intent(in)            :: v0     !< Root cell volume.
   integer(I4P),                                 intent(out)           :: stat   !< Status.
   character(:), allocatable,                    intent(out), optional :: errmsg !< Message.

   if (.not.self%fft%has_device_redundant()) then ! the host trip, as the parent's hook (not called through the parent
      ! component: its hooks would dispatch to the host's)
      call self%vec_update(self%lr(self%lbot)%v, to_device=.false.)
      self%root_buf = 0._R_P
      if (nc > 0) self%root_buf(1:nc) = self%lr(self%lbot)%v(1:nc) / v0
      call self%fft%solve_blocks(nbl=n, coords=self%root_coords, buf=self%root_buf, factor=1._R_P, gw=0_I4P, &
                                 out=self%root_out, stat=stat, errmsg=errmsg)
      if (stat /= ELISHA_OK) return
      if (nc > 0) self%le(self%lbot)%v(1:nc) = self%root_out(1:nc)
      call self%vec_update(self%le(self%lbot)%v, to_device=.true.)
      return
   endif
   if (nc > 0) call divide_dev(nc, size(self%lr(self%lbot)%v), self%lr(self%lbot)%v, v0, &
                               size(self%lt1(self%lbot)%v), self%lt1(self%lbot)%v)
   call self%fft%solve_redundant_device(nbl=n, coords=self%root_coords, nb=size(self%lt1(self%lbot)%v),              &
                                        buf=self%lt1(self%lbot)%v, nout=size(self%le(self%lbot)%v),                    &
                                        out=self%le(self%lbot)%v, factor=1._R_P, stat=stat, errmsg=errmsg)
   endsubroutine root_solve_oac

   subroutine stage_source_oac(self, buf, bufh, stat)
   !< `solve_packed(on_device)`: the device source copied, on the device, into a host array's device copy, then to the
   !< host (the client's `buf`, `intent(in)`, is not touched on either side).
   class(elisha_hierarchical_solver_oac_object), intent(inout) :: self    !< The solver.
   real(R_P),                                    intent(in)    :: buf(1:) !< Source (present).
   real(R_P), allocatable,                       intent(out)   :: bufh(:) !< Its host copy.
   integer(I4P),                                 intent(out)   :: stat    !< Status.
   integer(I4P)                                                :: i       !< Counter.

   if (.false.) stat = self%nr
   allocate(bufh(size(buf)))
   !$acc enter data create(bufh)
   !$acc parallel loop present(buf, bufh)
   do i=1, size(buf)
      bufh(i) = buf(i)
   enddo
   !$acc update self(bufh)
   !$acc exit data delete(bufh)
   stat = ELISHA_OK
   endsubroutine stage_source_oac

   subroutine stage_potential_oac(self, phi, to_host)
   !< `solve_packed(on_device)`: phi (present, contiguous) from the device to the host (`to_host`) or back.
   class(elisha_hierarchical_solver_oac_object), intent(inout) :: self         !< The solver.
   real(R_P),                                    intent(inout) :: phi(:,:,:,:) !< Potential (present).
   logical,                                      intent(in)    :: to_host      !< Direction.

   if (.false.) phi(1,1,1,1) = real(self%nr, R_P)
   if (to_host) then
      !$acc update self(phi)
   else
      !$acc update device(phi)
   endif
   endsubroutine stage_potential_oac

   subroutine solve_fcg_oac(self, mesh, buf, factor, phi, stat, errmsg, energy, ghost)
   !< The FCG solve: with the device assembly the device twin (`hierarchical_solve_fcg_device`: the right-hand side,
   !< phi and the energy by the hooks below), else ADAM's (`hierarchical_solve_fcg`: host assembly, staged); both with
   !< this self, so the FCG's vector hooks are the device's.
   class(elisha_hierarchical_solver_oac_object), intent(inout) :: self    !< The solver.
   type(elisha_mesh_object),       intent(in)            :: mesh    !< The mesh.
   real(R_P),                      intent(in)            :: buf(1:) !< Packed interior source.
   real(R_P),                      intent(in)            :: factor  !< Source factor.
   real(R_P),    intent(inout) :: phi(1-self%ngc:,1-self%ngc:,1-self%ngc:,1:) !< Potential [i,j,k,nb].
   integer(I4P),                   intent(out)           :: stat    !< Status.
   character(:), allocatable,      intent(out), optional :: errmsg  !< Message.
   real(R_P),                      intent(out), optional :: energy  !< sum_cells V f phi (all ranks).
   real(R_P),                      intent(in),  optional :: ghost(1:) !< The boundary value per boundary face.

   if (self%assembly == ELISHA_ASSEMBLY_DEVICE) then
      call hierarchical_solve_fcg_device(self, mesh, buf, factor, phi, stat, errmsg, energy, ghost)
   else
      call hierarchical_solve_fcg(self, mesh, buf, factor, phi, stat, errmsg, energy, ghost)
   endif
   endsubroutine solve_fcg_oac

   subroutine assemble_rhs_oac(self, buf, factor, n, fmax, vsum, stat, ghost)
   !< The right-hand side on the device (the device twin's hook, `ELISHA_ASSEMBLY_DEVICE`): `wf = factor V buf` (buf
   !< the caller's device array with `on_device`, else uploaded), the boundary terms `- w g` row by row in the host's
   !< face order, the singular mean removed (a device sum), `fmax = max |wf / V|`, the round-off floor's
   !< `max |factor buf|` (device maxima: exact), the stop tolerance as the host's; `u = 0` without a guess, on the
   !< device. Nothing is downloaded; host arrays upload buf (n) and the boundary values (nbnd). Collective.
   class(elisha_hierarchical_solver_oac_object), intent(inout) :: self      !< The solver.
   real(R_P),                      intent(in)            :: buf(1:)   !< Packed interior source.
   real(R_P),                      intent(in)            :: factor    !< Source factor.
   integer(I4P),                   intent(in)            :: n         !< Cells of this rank.
   real(R_P),                      intent(out)           :: fmax      !< max |f / V| over the ranks.
   real(R_P),                      intent(out)           :: vsum      !< sum V over the ranks.
   integer(I4P),                   intent(out)           :: stat      !< Status.
   real(R_P),                      intent(in),  optional :: ghost(1:) !< The boundary value per boundary face (host).
   real(R_P)                                             :: x1(1)     !< A scalar reduced in place.
   real(R_P)                                             :: mean, tol, braw !< Mean, stop tolerance, max |buf|.
   integer(I4P)                                          :: i         !< Counter.

   stat = ELISHA_OK
   ! sum V: the host's loop on the host's volumes (no transfer)
   x1 = 0._R_P
   do i=1, n
      x1(1) = x1(1) + self%vol(i) * 1._R_P
   enddo
   call self%comm%sum_real8(x1, stat) ; if (stat /= ELISHA_OK) return
   vsum = x1(1)
   if (.not.self%buf_device .and. n > 0) then
      self%dbuf(1:n) = buf(1:n)
      call update_dev(size(self%dbuf), self%dbuf, .true.)
   endif
   braw = 0._R_P
   if (n > 0) then
      if (self%buf_device) then
         call rhs_dev(n, size(buf), buf, factor, size(self%vol), self%vol, size(self%wf), self%wf)
         braw = absmax_dev(n, size(buf), buf)
      else
         call rhs_dev(n, size(self%dbuf), self%dbuf, factor, size(self%vol), self%vol, size(self%wf), self%wf)
         braw = absmax_dev(n, size(self%dbuf), self%dbuf)
      endif
   endif
   if (present(ghost) .and. self%nbnd > 0) then
      self%dghost(1:self%nbnd) = ghost(1:self%nbnd)
      call update_dev(size(self%dghost), self%dghost, .true.)
      if (self%nbrow > 0) call boundary_dev(self%nbrow, size(self%brow), self%brow, size(self%bptr), self%bptr,   &
                                            size(self%bidx), self%bidx, size(self%bnd_w), self%bnd_w,            &
                                            size(self%dghost), self%dghost, size(self%wf), self%wf)
   endif
   if (self%singular) then
      x1 = 0._R_P
      if (n > 0) x1(1) = sum_dev(n, size(self%wf), self%wf)
      call self%comm%sum_real8(x1, stat) ; if (stat /= ELISHA_OK) return
      mean = x1(1) / vsum
      if (n > 0) call axpy_dev(-mean, size(self%vol), self%vol, size(self%wf), self%wf, n)
   endif
   x1 = 0._R_P
   if (n > 0) x1(1) = relmax_dev(size(self%wf), self%wf, size(self%vol), self%vol, n)
   call self%comm%max_real8(x1, stat) ; if (stat /= ELISHA_OK) return
   fmax = x1(1)
   x1 = abs(factor) * braw
   call self%comm%max_real8(x1, stat) ; if (stat /= ELISHA_OK) return
   tol = self%tolerance
   if (fmax > 0._R_P) tol = max(tol, 100._R_P * epsilon(1._R_P) * x1(1) / fmax)
   self%tolerance_used = tol
   if (.not.self%has_guess) call fill_dev(size(self%u), self%u, 0._R_P, size(self%u))
   endsubroutine assemble_rhs_oac

   subroutine fill_potential_oac(self, mesh, buf, phi, n, stat, energy, ghost)
   !< phi and the energy on the device (the device twin's hook, `ELISHA_ASSEMBLY_DEVICE`): the ghost-fill operator
   !< on the device solution, each boundary row's value of its kind (the boundary values uploaded by
   !< `assemble_rhs_oac`), the energy `sum V buf u` (a device sum). With `on_device`, phi's interior and ghosts are
   !< written on the device (u downloaded only for the James step's host side); with host arrays, u and the final
   !< ghost values are downloaded and copied in, no arithmetic on the host: the two are bitwise equal. Collective.
   class(elisha_hierarchical_solver_oac_object), intent(inout) :: self    !< The solver.
   type(elisha_mesh_object),       intent(in)            :: mesh    !< The mesh.
   real(R_P),                      intent(in)            :: buf(1:) !< Packed interior source.
   real(R_P),    intent(inout) :: phi(1-self%ngc:,1-self%ngc:,1-self%ngc:,1:) !< Potential [i,j,k,nb].
   integer(I4P),                   intent(in)            :: n       !< Cells of this rank.
   integer(I4P),                   intent(out)           :: stat    !< Status.
   real(R_P),                      intent(out), optional :: energy  !< sum_cells V f phi (all ranks).
   real(R_P),                      intent(in),  optional :: ghost(1:) !< The boundary value per boundary face.
   real(R_P)                                             :: x1(1)   !< A scalar reduced in place.
   integer(I4P)                                          :: nrow    !< Ghost-fill rows.

   stat = ELISHA_OK
   nrow = self%g%nrow
   call self%apply(self%g, self%u, self%wg, stat) ; if (stat /= ELISHA_OK) return
   if (nrow > 0 .and. allocated(self%g_bnd)) &
      call ghost_values_dev(nrow, size(self%g_bnd), self%g_bnd, size(self%bnd_kind), self%bnd_kind,                &
                            size(self%bnd_gs), self%bnd_gs, size(self%dghost), self%dghost, size(self%wg), self%wg, &
                            present(ghost) .and. self%nbnd > 0, ELISHA_BC_DIRICHLET_GHOST)
   if (self%buf_device) then
      call scatter_dev(self%nijk, self%ngc, size(phi, dim=4), phi, mesh%nb, size(self%u), self%u, nrow,          &
                       size(self%g_dst, dim=2), self%g_dst, size(self%wg), self%wg)
      if (self%isolated) call update_dev(size(self%u), self%u, .false.) ! the James step's B^T u on the host
   else
      call update_dev(size(self%u), self%u, .false.)
      call update_dev(size(self%wg), self%wg, .false.)
      call scatter_host(self%nijk, self%ngc, size(phi, dim=4), phi, mesh%nb, size(self%u), self%u, nrow,         &
                        size(self%g_dst, dim=2), self%g_dst, size(self%wg), self%wg)
   endif
   if (present(energy)) then
      x1 = 0._R_P
      if (n > 0) then
         if (self%buf_device) then
            x1(1) = energy_dev(n, size(self%vol), self%vol, size(buf), buf, size(self%u), self%u)
         else
            x1(1) = energy_dev(n, size(self%vol), self%vol, size(self%dbuf), self%dbuf, size(self%u), self%u)
         endif
      endif
      call self%comm%sum_real8(x1, stat) ; if (stat /= ELISHA_OK) return
      energy = x1(1)
   endif
   endsubroutine fill_potential_oac

   ! private procedures
   subroutine divide_dev(n, nx, x, v0, ny, y)
   !< y(1:n) = x(1:n) / v0 on the device (the root's source, as the host path's `lr / v0`).
   integer(I4P), intent(in)    :: n, nx, ny !< Count, sizes.
   real(R_P),    intent(in)    :: x(nx)     !< Level residual (present).
   real(R_P),    intent(in)    :: v0        !< Root cell volume.
   real(R_P),    intent(inout) :: y(ny)     !< Root source (present).
   integer(I4P)                :: i         !< Counter.

   !$acc parallel loop present(x, y)
   do i=1, n
      y(i) = x(i) / v0
   enddo
   endsubroutine divide_dev

   subroutine halo_dev(self, m, x, stat)
   !< The halo of x for operator m into self%dhb: x(sidx) packed into self%dsb, exchanged on the device buffers.
   class(elisha_hierarchical_solver_oac_object), intent(inout) :: self  !< The solver.
   type(dcsr),                                   intent(in)    :: m     !< Operator.
   real(R_P),                                    intent(in)    :: x(1:) !< Local column-space vector (on the device).
   integer(I4P),                                 intent(out)   :: stat  !< Status of the exchange.

   stat = ELISHA_OK
   if (size(m%sidx) > 0) call pack_dev(size(m%sidx), m%sidx, size(x), x, size(self%dsb), self%dsb)
   if (self%nr > 1) call exchange_dev(m%plan, size(self%dsb), self%dsb, size(self%dhb), self%dhb, stat)
   endsubroutine halo_dev

   subroutine place_csr(m, attach)
   !< A CSR operator's arrays to (`attach`) or from the device (the exchange counts stay on the host).
   type(dcsr), intent(inout) :: m      !< Operator.
   logical,    intent(in)    :: attach !< Put, or release.

   call place_i(m%rowptr, attach) ; call place_i(m%col, attach) ; call place_r(m%val, attach)
   call place_i(m%sidx, attach)
   endsubroutine place_csr

   subroutine place_r(x, attach)
   !< A real array to (`attach`, copied in) or from the device (deleted); zero-size arrays stay on the host.
   real(R_P), intent(inout) :: x(:)   !< Array.
   logical,   intent(in)    :: attach !< Put, or release.

   if (size(x) == 0) return
   if (attach) then
      !$acc enter data copyin(x)
   else
      !$acc exit data delete(x)
   endif
   endsubroutine place_r

   subroutine place_i(x, attach)
   !< An integer array to (`attach`, copied in) or from the device (deleted); zero-size arrays stay on the host.
   integer(I4P), intent(inout) :: x(:)   !< Array.
   logical,      intent(in)    :: attach !< Put, or release.

   if (size(x) == 0) return
   if (attach) then
      !$acc enter data copyin(x)
   else
      !$acc exit data delete(x)
   endif
   endsubroutine place_i

   subroutine pack_dev(ns, sidx, nx, x, nb, sb)
   !< sb(i) = x(sidx(i)), i = 1..ns, on the device.
   integer(I4P), intent(in)    :: ns, nx, nb !< Sizes.
   integer(I4P), intent(in)    :: sidx(ns)   !< Local entries to send (present).
   real(R_P),    intent(in)    :: x(nx)      !< Vector (present).
   real(R_P),    intent(inout) :: sb(nb)     !< Send buffer (present).
   integer(I4P)                :: i          !< Counter.

   !$acc parallel loop present(sidx, x, sb)
   do i=1, ns
      sb(i) = x(sidx(i))
   enddo
   endsubroutine pack_dev

   subroutine exchange_dev(plan, ns, sb, nh, hb, stat)
   !< The exchange of the device buffers (GPU-aware MPI): the operator's halo plan (`MPI_Neighbor_alltoallv`, M7, audit
   !< B1) inside `host_data`, so its assumed-size dummies receive the device addresses and pass them to MPI (ADAM called
   !< `MPI_ALLTOALLV` here on `MPI_COMM_WORLD`).
   type(elisha_halo_plan), intent(in)    :: plan   !< The operator's halo plan.
   integer(I4P),           intent(in)    :: ns, nh !< Buffer sizes.
   real(R_P),              intent(in)    :: sb(ns) !< Send buffer (present).
   real(R_P),              intent(inout) :: hb(nh) !< Receive buffer (present).
   integer(I4P),           intent(out)   :: stat   !< Status of the exchange.

   !$acc host_data use_device(sb, hb)
   call plan%exchange_r(sb, hb, stat)
   !$acc end host_data
   endsubroutine exchange_dev

   subroutine spmv_dev(nrow, nloc, nnz, nx, ny, nh, rowptr, col, val, x, h, y)
   !< y = M x on the device, one row per thread, the entries in the host's order.
   integer(I4P), intent(in)    :: nrow, nloc, nnz, nx, ny, nh !< Sizes.
   integer(I4P), intent(in)    :: rowptr(nrow+1)              !< Row pointers (present).
   integer(I4P), intent(in)    :: col(nnz)                    !< Column slots (present).
   real(R_P),    intent(in)    :: val(nnz)                    !< Values (present).
   real(R_P),    intent(in)    :: x(nx)                       !< Local columns (present).
   real(R_P),    intent(in)    :: h(nh)                       !< Halo columns (present).
   real(R_P),    intent(inout) :: y(ny)                       !< Result (present).
   real(R_P)                   :: acc                         !< Accumulator.
   integer(I4P)                :: i, k                        !< Counters.

   !$acc parallel loop gang vector present(rowptr, col, val, x, h, y) private(acc)
   do i=1, nrow
      acc = 0._R_P
      !$acc loop seq
      do k=rowptr(i), rowptr(i+1)-1
         if (col(k) <= nloc) then
            acc = acc + val(k) * x(col(k))
         else
            acc = acc + val(k) * h(col(k) - nloc)
         endif
      enddo
      y(i) = acc
   enddo
   endsubroutine spmv_dev

   subroutine spmv_sten_dev(nrow, nloc, nnz, nx, ny, nh, rowptr, col, val, x, h, y, nl, lo, hi, nd, off, w)
   !< y = A_l x on the device, one row per thread (M7, audit C2): a block-interior row (its cell decoded from the row) as
   !< the 7-point stencil, its terms in its CSR entries' order (per active axis -w x_c, w x_low, -w x_c, w x_high);
   !< every other row from its CSR entries, as `spmv_dev`.
   integer(I4P), intent(in)    :: nrow, nloc, nnz, nx, ny, nh !< Sizes.
   integer(I4P), intent(in)    :: rowptr(nrow+1)              !< Row pointers (present).
   integer(I4P), intent(in)    :: col(nnz)                    !< Column slots (present).
   real(R_P),    intent(in)    :: val(nnz)                    !< Values (present).
   real(R_P),    intent(in)    :: x(nx)                       !< Local columns (present).
   real(R_P),    intent(in)    :: h(nh)                       !< Halo columns (present).
   real(R_P),    intent(inout) :: y(ny)                       !< Result (present).
   integer(I4P), intent(in)    :: nl(3), lo(3), hi(3)         !< Cells per block, the interior per axis.
   integer(I4P), intent(in)    :: nd                          !< Active axes.
   integer(I4P), intent(in)    :: off(3)                      !< Row offsets per active axis.
   real(R_P),    intent(in)    :: w(3)                        !< Face weights per active axis.
   real(R_P)                   :: acc                         !< Accumulator.
   integer(I4P)                :: i, k, rb, ii, jj, kk        !< Counters, the row in its block, the cell.
   integer(I4P)                :: n1, n2, nb, o1, o2, o3      !< Scalars of the stencil.
   real(R_P)                   :: w1, w2, w3                  !< Weights.
   integer(I4P)                :: l1, l2, l3, u1, u2, u3      !< The interior per axis.

   n1 = nl(1) ; n2 = nl(2) ; nb = nl(1) * nl(2) * nl(3)
   o1 = off(1) ; o2 = off(2) ; o3 = off(3) ; w1 = w(1) ; w2 = w(2) ; w3 = w(3)
   l1 = lo(1) ; l2 = lo(2) ; l3 = lo(3) ; u1 = hi(1) ; u2 = hi(2) ; u3 = hi(3)
   !$acc parallel loop gang vector present(rowptr, col, val, x, h, y) private(acc, rb, ii, jj, kk, k)
   do i=1, nrow
      rb = mod(i - 1, nb) ; ii = mod(rb, n1) + 1 ; jj = mod(rb / n1, n2) + 1 ; kk = rb / (n1 * n2) + 1
      acc = 0._R_P
      if (ii < l1 .or. ii > u1 .or. jj < l2 .or. jj > u2 .or. kk < l3 .or. kk > u3) then
         !$acc loop seq
         do k=rowptr(i), rowptr(i+1)-1
            if (col(k) <= nloc) then
               acc = acc + val(k) * x(col(k))
            else
               acc = acc + val(k) * h(col(k) - nloc)
            endif
         enddo
      else
         acc = acc + (-w1) * x(i)
         acc = acc + w1 * x(i - o1)
         acc = acc + (-w1) * x(i)
         acc = acc + w1 * x(i + o1)
         if (nd > 1) then
            acc = acc + (-w2) * x(i)
            acc = acc + w2 * x(i - o2)
            acc = acc + (-w2) * x(i)
            acc = acc + w2 * x(i + o2)
         endif
         if (nd > 2) then
            acc = acc + (-w3) * x(i)
            acc = acc + w3 * x(i - o3)
            acc = acc + (-w3) * x(i)
            acc = acc + w3 * x(i + o3)
         endif
      endif
      y(i) = acc
   enddo
   endsubroutine spmv_sten_dev

   subroutine sweep_sten_dev(nrow, nloc, nnz, ne, nr, nh, nd, colr, rowptr, col, val, diag, colour, rhs, h, e, nl, lo, hi, &
                             ns, off, w)
   !< One colour of a red-black sweep on the device (M7, audit C2): a block-interior row as the stencil, its neighbours in
   !< its CSR entries' order; every other row from its CSR entries, as `sweep_dev`.
   integer(I4P), intent(in)    :: nrow, nloc, nnz, ne, nr, nh, nd !< Sizes.
   integer(I4P), intent(in)    :: colr                            !< Colour of this pass.
   integer(I4P), intent(in)    :: rowptr(nrow+1)                  !< Row pointers (present).
   integer(I4P), intent(in)    :: col(nnz)                        !< Column slots (present).
   real(R_P),    intent(in)    :: val(nnz)                        !< Values (present).
   real(R_P),    intent(in)    :: diag(nd)                        !< Diagonal (present).
   integer(I4P), intent(in)    :: colour(nd)                      !< Colour of each row, 0 or 1 (present).
   real(R_P),    intent(in)    :: rhs(nr)                         !< Right-hand side (present).
   real(R_P),    intent(in)    :: h(nh)                           !< Halo of e (present).
   real(R_P),    intent(inout) :: e(ne)                           !< Correction (present).
   integer(I4P), intent(in)    :: nl(3), lo(3), hi(3)             !< Cells per block, the interior per axis.
   integer(I4P), intent(in)    :: ns                              !< Active axes.
   integer(I4P), intent(in)    :: off(3)                          !< Row offsets per active axis.
   real(R_P),    intent(in)    :: w(3)                            !< Face weights per active axis.
   real(R_P)                   :: acc                             !< Off-diagonal sum.
   integer(I4P)                :: i, k, rb, ii, jj, kk            !< Counters, the row in its block, the cell.
   integer(I4P)                :: n1, n2, nb, o1, o2, o3          !< Scalars of the stencil.
   real(R_P)                   :: w1, w2, w3                      !< Weights.
   integer(I4P)                :: l1, l2, l3, u1, u2, u3          !< The interior per axis.

   n1 = nl(1) ; n2 = nl(2) ; nb = nl(1) * nl(2) * nl(3)
   o1 = off(1) ; o2 = off(2) ; o3 = off(3) ; w1 = w(1) ; w2 = w(2) ; w3 = w(3)
   l1 = lo(1) ; l2 = lo(2) ; l3 = lo(3) ; u1 = hi(1) ; u2 = hi(2) ; u3 = hi(3)
   !$acc parallel loop gang vector present(rowptr, col, val, diag, colour, rhs, h, e) private(acc, rb, ii, jj, kk, k)
   do i=1, nrow
      if (colour(i) == colr) then
         rb = mod(i - 1, nb) ; ii = mod(rb, n1) + 1 ; jj = mod(rb / n1, n2) + 1 ; kk = rb / (n1 * n2) + 1
         acc = 0._R_P
         if (ii < l1 .or. ii > u1 .or. jj < l2 .or. jj > u2 .or. kk < l3 .or. kk > u3) then
            !$acc loop seq
            do k=rowptr(i), rowptr(i+1)-1
               if (col(k) == i) cycle
               if (col(k) <= nloc) then
                  acc = acc + val(k) * e(col(k))
               else
                  acc = acc + val(k) * h(col(k) - nloc)
               endif
            enddo
         else
            acc = acc + w1 * e(i - o1)
            acc = acc + w1 * e(i + o1)
            if (ns > 1) then
               acc = acc + w2 * e(i - o2)
               acc = acc + w2 * e(i + o2)
            endif
            if (ns > 2) then
               acc = acc + w3 * e(i - o3)
               acc = acc + w3 * e(i + o3)
            endif
         endif
         e(i) = (rhs(i) - acc) / diag(i)
      endif
   enddo
   endsubroutine sweep_sten_dev

   subroutine sweep_dev(nrow, nloc, nnz, ne, nr, nh, nd, colr, rowptr, col, val, diag, colour, rhs, h, e)
   !< One colour of a red-black Gauss-Seidel sweep on the device: e(i) = (rhs(i) - sum_{j /= i} a_ij e_j) / a_ii for the
   !< rows of colour colr, the off-diagonal entries in the host's order.
   integer(I4P), intent(in)    :: nrow, nloc, nnz, ne, nr, nh, nd !< Sizes.
   integer(I4P), intent(in)    :: colr                            !< Colour of this pass.
   integer(I4P), intent(in)    :: rowptr(nrow+1)                  !< Row pointers (present).
   integer(I4P), intent(in)    :: col(nnz)                        !< Column slots (present).
   real(R_P),    intent(in)    :: val(nnz)                        !< Values (present).
   real(R_P),    intent(in)    :: diag(nd)                        !< Diagonal (present).
   integer(I4P), intent(in)    :: colour(nd)                      !< Colour of each row, 0 or 1 (present).
   real(R_P),    intent(in)    :: rhs(nr)                         !< Right-hand side (present).
   real(R_P),    intent(in)    :: h(nh)                           !< Halo of e (present).
   real(R_P),    intent(inout) :: e(ne)                           !< Correction (present).
   real(R_P)                   :: acc                             !< Off-diagonal sum.
   integer(I4P)                :: i, k                            !< Counters.

   !$acc parallel loop gang vector present(rowptr, col, val, diag, colour, rhs, h, e) private(acc)
   do i=1, nrow
      if (colour(i) == colr) then
         acc = 0._R_P
         !$acc loop seq
         do k=rowptr(i), rowptr(i+1)-1
            if (col(k) == i) cycle
            if (col(k) <= nloc) then
               acc = acc + val(k) * e(col(k))
            else
               acc = acc + val(k) * h(col(k) - nloc)
            endif
         enddo
         e(i) = (rhs(i) - acc) / diag(i)
      endif
   enddo
   endsubroutine sweep_dev

   subroutine fill_dev(nx, x, a, n)
   !< x(1:n) = a on the device.
   integer(I4P), intent(in)    :: nx, n !< Sizes.
   real(R_P),    intent(inout) :: x(nx) !< Vector (present).
   real(R_P),    intent(in)    :: a     !< Value.
   integer(I4P)                :: i     !< Counter.

   !$acc parallel loop present(x)
   do i=1, n
      x(i) = a
   enddo
   endsubroutine fill_dev

   subroutine lincomb_dev(nx, x, a, ny, y, nw, w, n, comb)
   !< w(1:n) = x(1:n) + a y(1:n) (comb) or w(1:n) = x(1:n) on the device.
   integer(I4P), intent(in)    :: nx, ny, nw, n !< Sizes.
   real(R_P),    intent(in)    :: x(nx)         !< First vector (present).
   real(R_P),    intent(in)    :: a             !< Coefficient.
   real(R_P),    intent(in)    :: y(ny)         !< Second vector (present).
   real(R_P),    intent(inout) :: w(nw)         !< Result (present).
   logical,      intent(in)    :: comb          !< Combine, or copy x.
   integer(I4P)                :: i             !< Counter.

   if (comb) then
      !$acc parallel loop present(x, y, w)
      do i=1, n
         w(i) = x(i) + a * y(i)
      enddo
   else
      !$acc parallel loop present(x, w)
      do i=1, n
         w(i) = x(i)
      enddo
   endif
   endsubroutine lincomb_dev

   subroutine axpy_dev(a, nx, x, ny, y, n)
   !< y(1:n) = y(1:n) + a x(1:n) on the device.
   real(R_P),    intent(in)    :: a         !< Coefficient.
   integer(I4P), intent(in)    :: nx, ny, n !< Sizes.
   real(R_P),    intent(in)    :: x(nx)     !< Vector (present).
   real(R_P),    intent(inout) :: y(ny)     !< Updated vector (present).
   integer(I4P)                :: i         !< Counter.

   !$acc parallel loop present(x, y)
   do i=1, n
      y(i) = y(i) + a * x(i)
   enddo
   endsubroutine axpy_dev

   function dot_dev(nx, x, ny, y, n) result(d)
   !< sum_{i<=n} x_i y_i on the device.
   integer(I4P), intent(in) :: nx, ny, n !< Sizes.
   real(R_P),    intent(in) :: x(nx)     !< First vector (present).
   real(R_P),    intent(in) :: y(ny)     !< Second vector (present).
   real(R_P)                :: d         !< Result.
   integer(I4P)             :: i         !< Counter.

   d = 0._R_P
   !$acc parallel loop present(x, y) reduction(+:d)
   do i=1, n
      d = d + x(i) * y(i)
   enddo
   endfunction dot_dev

   function relmax_dev(nr, r, nv, vol, n) result(rm)
   !< max_{i<=n} |r_i / vol_i| on the device.
   integer(I4P), intent(in) :: nr, nv, n !< Sizes.
   real(R_P),    intent(in) :: r(nr)     !< Residual (present).
   real(R_P),    intent(in) :: vol(nv)   !< Cell volumes (present).
   real(R_P)                :: rm        !< Result.
   integer(I4P)             :: i         !< Counter.

   rm = 0._R_P
   !$acc parallel loop present(r, vol) reduction(max:rm)
   do i=1, n
      rm = max(rm, abs(r(i) / vol(i)))
   enddo
   endfunction relmax_dev

   subroutine permute_dev(nx, x, nix, ix, ny, y, niy, iy, n)
   !< y(iy(i)) = x(ix(i)), i = 1..n, on the device.
   integer(I4P), intent(in)    :: nx, nix, ny, niy, n !< Sizes.
   real(R_P),    intent(in)    :: x(nx)               !< Source (present).
   integer(I4P), intent(in)    :: ix(nix)             !< Source entries (present).
   real(R_P),    intent(inout) :: y(ny)               !< Destination (present).
   integer(I4P), intent(in)    :: iy(niy)             !< Destination entries (present).
   integer(I4P)                :: i                   !< Counter.

   !$acc parallel loop present(x, ix, y, iy)
   do i=1, n
      y(iy(i)) = x(ix(i))
   enddo
   endsubroutine permute_dev

   subroutine update_dev(nx, x, to_device)
   !< x to (`to_device`) or from the device.
   integer(I4P), intent(in)    :: nx        !< Size.
   real(R_P),    intent(inout) :: x(nx)     !< Vector (present).
   logical,      intent(in)    :: to_device !< Direction.

   if (to_device) then
      !$acc update device(x)
   else
      !$acc update self(x)
   endif
   endsubroutine update_dev

   ! the device assembly (ELISHA notes "device assembly")
   subroutine group_boundary_rows(self)
   !< The boundary faces grouped by the row they bound, each row's faces in increasing face order (the host's order of
   !< `wf(bnd_row(i)) - bnd_w(i) g(i)`, i = 1..nbnd: a corner cell's terms are subtracted in the same sequence); and
   !< the device assembly's work vectors, the source of a host-array solve and the boundary values.
   class(elisha_hierarchical_solver_oac_object), intent(inout) :: self !< The solver.
   integer(I4P), allocatable :: cnt(:), first(:) !< Faces per row, the position of each row in brow.
   integer(I4P)              :: i, r, q, nrows    !< Counters, rows of the local leaves.

   if (allocated(self%dbuf)) deallocate(self%dbuf, self%dghost, self%brow, self%bptr, self%bidx)
   allocate(self%dbuf(size(self%u)), self%dghost(max(self%nbnd,1)))
   self%dbuf = 0._R_P ; self%dghost = 0._R_P
   nrows = size(self%u)
   allocate(cnt(nrows), first(nrows))
   cnt = 0 ; first = 0
   do i=1, self%nbnd
      cnt(self%bnd_row(i)) = cnt(self%bnd_row(i)) + 1
   enddo
   self%nbrow = count(cnt > 0)
   allocate(self%brow(max(self%nbrow,1)), self%bptr(self%nbrow+1), self%bidx(max(self%nbnd,1)))
   self%brow = 0 ; self%bidx = 0
   q = 0
   do r=1, nrows ! the rows in increasing order (any order is correct: a row's terms are independent of another's)
      if (cnt(r) == 0) cycle
      q = q + 1
      self%brow(q) = r
      first(r) = q
   enddo
   self%bptr(1) = 1
   do q=1, self%nbrow
      self%bptr(q+1) = self%bptr(q) + cnt(self%brow(q))
   enddo
   cnt = 0
   do i=1, self%nbnd ! increasing face order within each row
      r = self%bnd_row(i)
      q = first(r)
      self%bidx(self%bptr(q) + cnt(r)) = i
      cnt(r) = cnt(r) + 1
   enddo
   endsubroutine group_boundary_rows

   subroutine place_i2(x, attach)
   !< A rank-2 integer array to (`attach`, copied in) or from the device (deleted); zero-size arrays stay on the host.
   integer(I4P), intent(inout) :: x(:,:) !< Array.
   logical,      intent(in)    :: attach !< Put, or release.

   if (size(x) == 0) return
   if (attach) then
      !$acc enter data copyin(x)
   else
      !$acc exit data delete(x)
   endif
   endsubroutine place_i2

   subroutine rhs_dev(n, nb, b, factor, nv, vol, nf, f)
   !< f(1:n) = factor vol(1:n) b(1:n) on the device (the host's expression and order).
   integer(I4P), intent(in)    :: n, nb, nv, nf !< Count, sizes.
   real(R_P),    intent(in)    :: b(nb)         !< Source (present).
   real(R_P),    intent(in)    :: factor        !< Source factor.
   real(R_P),    intent(in)    :: vol(nv)       !< Cell volumes (present).
   real(R_P),    intent(inout) :: f(nf)         !< Right-hand side (present).
   integer(I4P)                :: i             !< Counter.

   !$acc parallel loop present(b, vol, f)
   do i=1, n
      f(i) = factor * vol(i) * b(i)
   enddo
   endsubroutine rhs_dev

   function absmax_dev(n, nb, b) result(m)
   !< max_{i<=n} |b_i| on the device (exact: a maximum does not depend on the order).
   integer(I4P), intent(in) :: n, nb !< Count, size.
   real(R_P),    intent(in) :: b(nb) !< Vector (present).
   real(R_P)                :: m     !< Result.
   integer(I4P)             :: i     !< Counter.

   m = 0._R_P
   !$acc parallel loop present(b) reduction(max:m)
   do i=1, n
      m = max(m, abs(b(i)))
   enddo
   endfunction absmax_dev

   function sum_dev(n, nx, x) result(s)
   !< sum_{i<=n} x_i on the device.
   integer(I4P), intent(in) :: n, nx !< Count, size.
   real(R_P),    intent(in) :: x(nx) !< Vector (present).
   real(R_P)                :: s     !< Result.
   integer(I4P)             :: i     !< Counter.

   s = 0._R_P
   !$acc parallel loop present(x) reduction(+:s)
   do i=1, n
      s = s + x(i)
   enddo
   endfunction sum_dev

   subroutine boundary_dev(nr, nrw, brow, np, bptr, ni, bidx, nw, bw, ng, g, nf, f)
   !< The boundary terms, one row per thread, its faces in face order: f(r) = f(r) - bw(i) g(i).
   integer(I4P), intent(in)    :: nr, nrw, np, ni, nw, ng, nf !< Rows, sizes.
   integer(I4P), intent(in)    :: brow(nrw)  !< The rows (present).
   integer(I4P), intent(in)    :: bptr(np)   !< Pointers (present).
   integer(I4P), intent(in)    :: bidx(ni)   !< Faces (present).
   real(R_P),    intent(in)    :: bw(nw)     !< Face weights in B (present).
   real(R_P),    intent(in)    :: g(ng)      !< Boundary values (present).
   real(R_P),    intent(inout) :: f(nf)      !< Right-hand side (present).
   integer(I4P)                :: q, p, r, i !< Counters.

   !$acc parallel loop present(brow, bptr, bidx, bw, g, f)
   do q=1, nr
      r = brow(q)
      !$acc loop seq
      do p=bptr(q), bptr(q+1)-1
         i = bidx(p)
         f(r) = f(r) - bw(i) * g(i)
      enddo
   enddo
   endsubroutine boundary_dev

   subroutine ghost_values_dev(nrow, ngb, g_bnd, nk, kind, ns, gs, ng, g, nv, gv, has_ghost, dg_kind)
   !< Each ghost-fill row on a domain face: the ghost of its kind (the host's rule): ghost-Dirichlet, the value (0
   !< without values); another kind with values, gv + gs g.
   integer(I4P), intent(in)    :: nrow, ngb, nk, ns, ng, nv !< Rows, sizes.
   integer(I4P), intent(in)    :: g_bnd(ngb) !< Row -> boundary face, 0 inside (present).
   integer(I4P), intent(in)    :: kind(nk)   !< Face kinds (present).
   real(R_P),    intent(in)    :: gs(ns)     !< Value weights in the ghost fill (present).
   real(R_P),    intent(in)    :: g(ng)      !< Boundary values (present).
   real(R_P),    intent(inout) :: gv(nv)     !< Ghost values (present).
   logical,      intent(in)    :: has_ghost  !< Boundary values given.
   integer(I4P), intent(in)    :: dg_kind    !< ELISHA_BC_DIRICHLET_GHOST.
   integer(I4P)                :: row, fb    !< Counters.

   !$acc parallel loop present(g_bnd, kind, gs, g, gv)
   do row=1, nrow
      fb = g_bnd(row)
      if (fb > 0) then
         if (kind(fb) == dg_kind) then
            gv(row) = 0._R_P
            if (has_ghost) gv(row) = g(fb)
         elseif (has_ghost) then
            gv(row) = gv(row) + gs(fb) * g(fb)
         endif
      endif
   enddo
   endsubroutine ghost_values_dev

   subroutine scatter_dev(nl, ng, nbp, phi, nb, nu, u, nrow, nd, dst, nv, gv)
   !< phi on the device: the interior from u (block-major, i fastest), each ghost-fill row's value at its cell, the
   !< deeper layers copying the first along the normal (the host's copies).
   integer(I4P), intent(in)    :: nl(3), ng, nbp, nb, nu, nrow, nd, nv !< Cells per block, ghosts, sizes.
   real(R_P),    intent(inout) :: phi(1-ng:nl(1)+ng,1-ng:nl(2)+ng,1-ng:nl(3)+ng,nbp) !< Potential (present).
   real(R_P),    intent(in)    :: u(nu)      !< Solution (present).
   integer(I4P), intent(in)    :: dst(4,nd)  !< Ghost-fill destinations (present).
   real(R_P),    intent(in)    :: gv(nv)     !< Ghost values (present).
   integer(I4P)                :: b, i, j, k, row, l !< Counters.

   !$acc parallel loop collapse(4) present(phi, u)
   do b=1, nb
      do k=1, nl(3)
         do j=1, nl(2)
            do i=1, nl(1)
               phi(i,j,k,b) = u((b - 1) * nl(1) * nl(2) * nl(3) + i + (j - 1) * nl(1) + (k - 1) * nl(1) * nl(2))
            enddo
         enddo
      enddo
   enddo
   !$acc parallel loop present(phi, dst, gv)
   do row=1, nrow
      b = dst(4,row) ; i = dst(1,row) ; j = dst(2,row) ; k = dst(3,row)
      phi(i,j,k,b) = gv(row)
      !$acc loop seq
      do l=2, ng
         if (i < 1)     phi(1-l,j,k,b) = gv(row)
         if (i > nl(1)) phi(nl(1)+l,j,k,b) = gv(row)
         if (j < 1)     phi(i,1-l,k,b) = gv(row)
         if (j > nl(2)) phi(i,nl(2)+l,k,b) = gv(row)
         if (k < 1)     phi(i,j,1-l,b) = gv(row)
         if (k > nl(3)) phi(i,j,nl(3)+l,b) = gv(row)
      enddo
   enddo
   endsubroutine scatter_dev

   subroutine scatter_host(nl, ng, nbp, phi, nb, nu, u, nrow, nd, dst, nv, gv)
   !< scatter_dev's copies on the host (host arrays): the same values, no arithmetic.
   integer(I4P), intent(in)    :: nl(3), ng, nbp, nb, nu, nrow, nd, nv !< Cells per block, ghosts, sizes.
   real(R_P),    intent(inout) :: phi(1-ng:nl(1)+ng,1-ng:nl(2)+ng,1-ng:nl(3)+ng,nbp) !< Potential (host).
   real(R_P),    intent(in)    :: u(nu)      !< Solution (host copy).
   integer(I4P), intent(in)    :: dst(4,nd)  !< Ghost-fill destinations.
   real(R_P),    intent(in)    :: gv(nv)     !< Ghost values (host copy).
   integer(I4P)                :: b, i, j, k, row, l, c !< Counters.

   c = 0
   do b=1, nb
      do k=1, nl(3)
         do j=1, nl(2)
            do i=1, nl(1)
               c = c + 1
               phi(i,j,k,b) = u(c)
            enddo
         enddo
      enddo
   enddo
   do row=1, nrow
      b = dst(4,row) ; i = dst(1,row) ; j = dst(2,row) ; k = dst(3,row)
      phi(i,j,k,b) = gv(row)
      do l=2, ng
         if (i < 1)     phi(1-l,j,k,b) = gv(row)
         if (i > nl(1)) phi(nl(1)+l,j,k,b) = gv(row)
         if (j < 1)     phi(i,1-l,k,b) = gv(row)
         if (j > nl(2)) phi(i,nl(2)+l,k,b) = gv(row)
         if (k < 1)     phi(i,j,1-l,b) = gv(row)
         if (k > nl(3)) phi(i,j,nl(3)+l,b) = gv(row)
      enddo
   enddo
   endsubroutine scatter_host

   function energy_dev(n, nv, vol, nb, b, nu, u) result(e)
   !< sum_{i<=n} vol_i b_i u_i on the device (the host's product order).
   integer(I4P), intent(in) :: n, nv, nb, nu !< Count, sizes.
   real(R_P),    intent(in) :: vol(nv)       !< Cell volumes (present).
   real(R_P),    intent(in) :: b(nb)         !< Source (present).
   real(R_P),    intent(in) :: u(nu)         !< Solution (present).
   real(R_P)                :: e             !< Result.
   integer(I4P)             :: i             !< Counter.

   e = 0._R_P
   !$acc parallel loop present(vol, b, u) reduction(+:e)
   do i=1, n
      e = e + vol(i) * b(i) * u(i)
   enddo
   endfunction energy_dev
endmodule elisha_hierarchical_solver_oac
