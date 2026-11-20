!< ELISHA, the spectral solver's device backend: cuFFT and OpenACC on NVIDIA GPUs.
module elisha_spectral_solver_cufft
!< ELISHA, the spectral solver's device backend: cuFFT and OpenACC on NVIDIA GPUs.
!<
!< **Provenance: ADAM's `adam_fnl_poisson_fft_object` (ADAM 669d0bdd), plumbing changed, arithmetic not.** It extends
!< `elisha_spectral_solver_object` (ADAM's `poisson_fft_object` ported) as ADAM's extends its host object, and overrides
!< the same backend hooks. The device code is ADAM's: every OpenACC construct and clause, every kernel's loops and
!< their order, every expression, the cuFFT plans (sizes, types, batch) on the synchronous OpenACC queue's stream, the
!< data placement (`enter`/`exit data`, `update`, the `data` regions) and the GPU-aware `MPI_ALLTOALLV` on the device
!< buffers; so the potential is meant to be ADAM's device potential bitwise, on the same GPU and compiler
!< (`src/tests/device/elisha_test_device_parity_mpi`, against the verbatim ADAM twin). What changed:
!< - `mpih%myrank`, `mpih%procs_number` are `self%comm%rank`, `self%comm%nranks`;
!< - the transpose's exchange is `elisha_comm_object%alltoallv_c` (ADAM: `MPI_ALLTOALLV` on `MPI_COMM_WORLD`), called
!<   inside the same `host_data use_device(sb, rb)` region with the same buffers: its dummies are assumed-size
!<   (`x(*)`), so the device addresses of the contiguous explicit-shape buffers pass by reference, untouched, to its
!<   `MPI_ALLTOALLV` (the `mpi` module's choice buffers), on the solver's duplicated communicator; ADAM ignored the
!<   MPI error code, here it is the status (`ELISHA_ERR_MPI`);
!< - `mpih%error_stop` is a status: the hooks that have `stat` in ELISHA (`make_plans`, `transform`,
!<   `transform_kernel`, `slab_make_plans`, `slab_planes`, `slab_lines`, `slab_transpose`) return
!<   `ELISHA_ERR_FFT_PLAN` when cuFFT fails (the private procedures return the cuFFT status where ADAM stopped, and
!<   run on as ADAM's success path does: no construct moved); the others had no failure path with cuFFT;
!< - ADAM's builds without cuFFT stopped at the plans (an AMD build); this module needs `-D_CUFFT` (and `-acc`,
!<   `-cudalib=cufft`): without it the build stops (`#error`), so no hook carries ADAM's `#else` branches;
!< - one M7 change (audit C5), not changing a bit: the setup-time constants the kernels read are device-resident,
!<   placed once with the plans (`enter data copyin`) and released by the plan hooks (`exit data delete`) before the
!<   base `destroy` deallocates them: the eigenvalue parts `lam1`, `lam2`, `lam3` (`make_plans` or `slab_make_plans`,
!<   released by `destroy_plans`), and the transpose's row ranges `yr` and row owners `yown` (`slab_make_plans`,
!<   released by `slab_destroy_plans`). The kernels name them `present`; ADAM copied them in at every call
!<   (`copyin(lam1, lam2, lam3)` in the rank-0 solve and the slab division, `copyin(yr, yown)` in the transpose's pack
!<   and unpack). A setup after a regrid destroys first, so the copies are never doubled.
!< The `R8P` of ADAM is ELISHA's `R_P` (R8P, `elisha_kind_parameters`).
!<
!< ADAM's description, unchanged: the gather to rank 0, the placement and the scatter of `phi` with its ghosts are the
!< host ones; the rank-0 (and redundant) solve runs on the GPU through cuFFT: the global source to the device, a D2Z
!< transform, the division by the eigenvalues of `L_h` (the same host-computed `lam1..3`), a Z2D transform, the
!< potential back. cuFFT runs on the stream of the synchronous OpenACC queue, so the transforms and the kernels between
!< them are ordered. Isolated: the transforms are the padded ones (`m`), the kernel's spectrum `khat` computed once on
!< the device (`transform_kernel`) and kept there (`enter data`, released by `destroy_plans`); the solve multiplies by
!< it. The distributed slab transform keeps the host redistribution and return of the block planes; the transform
!< runs on each rank's GPU, its arrays resident there (`slab_r`, `slab_c`, `pen`, `khat_pen`, the transpose buffers:
!< `enter data` with the plans): `slab_r` up, the local planes (one D2Z or Z2D plan for one plane, executed on every
!< plane in turn), the transpose to z-lines (packed and unpacked on the device, the exchange on the device buffers),
!< the local z-lines (one Z2Z plan for one row of h1 lines, on every row in turn), the spectral solve, back, and
!< `slab_r` down. The arithmetic of a value does not depend on the rank count. A multigrid root on cuFFT has at most
!< 32^3 cells (`root_cells_max`).
#ifndef _CUFFT
#error "elisha_spectral_solver_cufft needs cuFFT: build with -D_CUFFT -acc -cudalib=cufft (fobos modes *-nvf-oac)"
#endif
use, intrinsic :: iso_c_binding
use cufft
use openacc
use penf, only : I4P, I8P, str
use elisha_kind_parameters, only : R_P
use elisha_comm, only : elisha_comm_object
use elisha_spectral_solver, only : elisha_spectral_solver_object
use elisha_status, only : ELISHA_OK, ELISHA_ERR_FFT_PLAN, elisha_set_error
implicit none
private
public :: elisha_spectral_solver_cufft_object

type, extends(elisha_spectral_solver_object) :: elisha_spectral_solver_cufft_object
   !< The spectral solver on cuFFT and OpenACC (see the module).
   integer(c_int) :: plan_d2z=0_c_int  !< cuFFT forward plan (rank 0).
   integer(c_int) :: plan_z2d=0_c_int  !< cuFFT backward plan (rank 0).
   logical        :: has_plans=.false. !< The cuFFT plans exist.
   logical        :: khat_on_device=.false. !< `khat` is on the device (isolated).
   integer(c_int) :: plan_sd2z=0_c_int  !< cuFFT slab plan: one plane, D2Z (ADAM #84 E5d).
   integer(c_int) :: plan_sz2d=0_c_int  !< cuFFT slab plan: one plane, Z2D.
   integer(c_int) :: plan_sz2z=0_c_int  !< cuFFT slab plan: one row of z-lines, Z2Z.
   logical        :: has_slab_plans=.false. !< The cuFFT slab plans exist.
   logical        :: slab_on_device=.false. !< The slab arrays are on the device.
   logical        :: lam_on_device=.false.  !< lam1, lam2, lam3 are on the device (M7, audit C5).
   logical        :: maps_on_device=.false. !< yr, yown are on the device (M7, audit C5).
   integer(I4P), allocatable :: yown(:)     !< Owner rank of each y-row [0:m(2)-1].
   complex(R_P), allocatable :: t_sb(:)     !< Device: the transpose's send buffer.
   complex(R_P), allocatable :: t_rb(:)     !< Device: the transpose's receive buffer.
   contains
      ! backend hooks
      procedure, pass(self) :: destroy_plans => destroy_plans_cufft !< Destroy the cuFFT plans.
      procedure, pass(self) :: make_plans    => make_plans_cufft    !< Make the cuFFT plans (rank 0).
      procedure, pass(self) :: transform     => transform_cufft     !< The spectral solve on the GPU (rank 0).
      procedure, pass(self) :: transform_kernel => transform_kernel_cufft !< The kernel's spectrum on the GPU.
      procedure, nopass     :: root_cells_max => root_cells_max_cufft   !< 32^3: the device smooths, the host transforms.
      procedure, pass(self) :: slab_make_plans    => slab_make_plans_cufft    !< The cuFFT slab plans (E5d).
      procedure, pass(self) :: slab_destroy_plans => slab_destroy_plans_cufft !< Destroy them.
      procedure, pass(self) :: slab_planes        => slab_planes_cufft        !< Every local plane on the GPU.
      procedure, pass(self) :: slab_lines         => slab_lines_cufft         !< Every local z-line on the GPU.
      procedure, pass(self) :: slab_transpose     => slab_transpose_cufft     !< The transpose on the GPU.
      procedure, pass(self) :: slab_spectral      => slab_spectral_cufft      !< The spectral solve on the GPU.
      procedure, pass(self) :: slab_keep_kernel   => slab_keep_kernel_cufft   !< khat_pen on the GPU.
      procedure, pass(self) :: slab_update        => slab_update_cufft        !< slab_r to and from the GPU.
endtype elisha_spectral_solver_cufft_object

contains
   ! backend hooks
   subroutine destroy_plans_cufft(self)
   !< Destroy the cuFFT plans.
   class(elisha_spectral_solver_cufft_object), intent(inout) :: self !< The solver.
   integer(c_int)                                            :: ierr !< cuFFT status.

   ierr = 0_c_int
   if (self%has_plans) then
      ierr = cufftDestroy(self%plan_d2z)
      ierr = cufftDestroy(self%plan_z2d)
   endif
   if (self%khat_on_device) then
      !$acc exit data delete(self%khat)
   endif
   if (self%lam_on_device) then
      !$acc exit data delete(self%lam1, self%lam2, self%lam3)
   endif
   self%has_plans = .false.
   self%khat_on_device = .false.
   self%lam_on_device = .false.
   endsubroutine destroy_plans_cufft

   subroutine make_plans_cufft(self, stat, errmsg)
   !< Make the cuFFT plans (D2Z and Z2D, cuFFT is row-major: the slowest dimension first) on the synchronous queue.
   class(elisha_spectral_solver_cufft_object), intent(inout)         :: self   !< The solver.
   integer(I4P),                               intent(out)           :: stat   !< Status.
   character(:), allocatable,                  intent(out), optional :: errmsg !< Message.
   integer(c_int)                                                    :: ierr   !< cuFFT status.

   stat = ELISHA_OK
   ierr = 0_c_int
   ierr = cufftPlan3d(self%plan_d2z, int(self%m(3), c_int), int(self%m(2), c_int), int(self%m(1), c_int), CUFFT_D2Z)
   if (ierr /= CUFFT_SUCCESS) then
      call elisha_set_error(stat, errmsg, ELISHA_ERR_FFT_PLAN, 'elisha_spectral_solver_cufft_object%make_plans: '// &
                            'cufftPlan3d D2Z failed, status '//trim(str(int(ierr, I4P), .true.)))
      return
   endif
   ierr = cufftPlan3d(self%plan_z2d, int(self%m(3), c_int), int(self%m(2), c_int), int(self%m(1), c_int), CUFFT_Z2D)
   if (ierr /= CUFFT_SUCCESS) then
      call elisha_set_error(stat, errmsg, ELISHA_ERR_FFT_PLAN, 'elisha_spectral_solver_cufft_object%make_plans: '// &
                            'cufftPlan3d Z2D failed, status '//trim(str(int(ierr, I4P), .true.)))
      return
   endif
   ierr = cufftSetStream(self%plan_d2z, acc_get_cuda_stream(acc_async_sync))
   ierr = ierr + cufftSetStream(self%plan_z2d, acc_get_cuda_stream(acc_async_sync))
   if (ierr /= CUFFT_SUCCESS) then
      call elisha_set_error(stat, errmsg, ELISHA_ERR_FFT_PLAN, 'elisha_spectral_solver_cufft_object%make_plans: '// &
                            'cufftSetStream failed')
      return
   endif
   self%has_plans = .true.
   call place_lam(self)
   endsubroutine make_plans_cufft

   subroutine transform_cufft(self, factor, stat, errmsg)
   !< Rank 0: `f_global <- factor * L_h^-1 f_global` on the GPU, the modes of zero eigenvalue (the mean) set to zero.
   class(elisha_spectral_solver_cufft_object), intent(inout)         :: self   !< The solver.
   real(R_P),                                  intent(in)            :: factor !< Source factor.
   integer(I4P),                               intent(out)           :: stat   !< Status.
   character(:), allocatable,                  intent(out), optional :: errmsg !< Message.
   integer(c_int)                                                    :: ierr   !< cuFFT status.

   stat = ELISHA_OK
   if (self%isolated) then
      call spectral_convolve_dev(plan_d2z=self%plan_d2z, plan_z2d=self%plan_z2d, m1=self%m(1), m2=self%m(2), &
                                 m3=self%m(3), factor=factor, khat=self%khat, f=self%f_global, f_hat=self%f_hat, &
                                 ierr=ierr)
   else
      call spectral_solve_dev(plan_d2z=self%plan_d2z, plan_z2d=self%plan_z2d,                     &
                              n1=self%nt(1), n2=self%nt(2), n3=self%nt(3),                         &
                              scale=factor / real(product(self%nt), R_P),                          &
                              lam1=self%lam1, lam2=self%lam2, lam3=self%lam3, f=self%f_global, f_hat=self%f_hat, &
                              ierr=ierr)
   endif
   if (ierr /= CUFFT_SUCCESS) call elisha_set_error(stat, errmsg, ELISHA_ERR_FFT_PLAN, &
      'elisha_spectral_solver_cufft_object%transform: a cuFFT transform failed, status '//trim(str(int(ierr, I4P), .true.)))
   endsubroutine transform_cufft

   subroutine transform_kernel_cufft(self, stat, errmsg)
   !< Rank 0, isolated: `khat <- FFT(f_global) / prod(m)` on the GPU (real: the kernel is even); `khat` stays on the
   !< device for every solve.
   class(elisha_spectral_solver_cufft_object), intent(inout)         :: self   !< The solver.
   integer(I4P),                               intent(out)           :: stat   !< Status.
   character(:), allocatable,                  intent(out), optional :: errmsg !< Message.
   integer(c_int)                                                    :: ierr   !< cuFFT status.

   stat = ELISHA_OK
   call kernel_spectrum_dev(plan_d2z=self%plan_d2z, m1=self%m(1), m2=self%m(2), m3=self%m(3), f=self%f_global, &
                            f_hat=self%f_hat, ierr=ierr)
   if (ierr /= CUFFT_SUCCESS) then
      call elisha_set_error(stat, errmsg, ELISHA_ERR_FFT_PLAN, 'elisha_spectral_solver_cufft_object%transform_kernel: '// &
                            'cufftExecD2Z failed (kernel), status '//trim(str(int(ierr, I4P), .true.)))
      return
   endif
   self%khat = real(self%f_hat, R_P) / real(product(self%m), R_P)
   !$acc enter data copyin(self%khat)
   self%khat_on_device = .true.
   endsubroutine transform_kernel_cufft

   pure function root_cells_max_cufft(growth) result(n)
   !< The most cells of a multigrid root on cuFFT (ADAM #84 E6, E6b): 32^3 whatever the transform's growth. The
   !< multigrid smooths on the device and the root's transform goes through the host's redistribution, so a small root
   !< wins (ADAM, medooza FNL, 3.9 M AMR cells, 1 / 2 / 4 GPUs: periodic 0.33 / 0.19 / 0.12 s with a 32^3 root, 0.80 /
   !< 0.45 / 0.37 s with 128^3; for the isolated root 16^3 gains only 3-10% over 32^3).
   integer(I4P), intent(in) :: growth !< Transform cells per root cell (unused: the cap is in cells).
   integer(I4P)             :: n      !< Cells.

   n = 32768_I4P
   if (.false.) n = growth
   endfunction root_cells_max_cufft

   subroutine slab_make_plans_cufft(self, stat, errmsg)
   !< The cuFFT slab plans (E5d): one plane D2Z and Z2D, one row of z-lines Z2Z (h1 lines of m(3) cells, contiguous),
   !< on the synchronous OpenACC queue's stream; the slab arrays and the transpose buffers on the device.
   class(elisha_spectral_solver_cufft_object), intent(inout)         :: self   !< The solver.
   integer(I4P),                               intent(out)           :: stat   !< Status.
   character(:), allocatable,                  intent(out), optional :: errmsg !< Message.
   integer(c_int)                                                    :: ierr   !< cuFFT status.
   integer(I4P)                                                      :: r, me, nb !< Counter, rank, buffer size.

   stat = ELISHA_OK
   ierr = 0_c_int
   ierr = cufftPlan2d(self%plan_sd2z, int(self%m(2), c_int), int(self%m(1), c_int), CUFFT_D2Z)
   ierr = ierr + cufftPlan2d(self%plan_sz2d, int(self%m(2), c_int), int(self%m(1), c_int), CUFFT_Z2D)
   ierr = ierr + cufftPlanMany(self%plan_sz2z, 1_c_int, [int(self%m(3), c_int)], [int(self%m(3), c_int)], 1_c_int, &
                               int(self%m(3), c_int), [int(self%m(3), c_int)], 1_c_int, int(self%m(3), c_int),    &
                               CUFFT_Z2Z, int(self%h1, c_int))
   if (ierr /= CUFFT_SUCCESS) then
      call elisha_set_error(stat, errmsg, ELISHA_ERR_FFT_PLAN, 'elisha_spectral_solver_cufft_object%slab_make_plans: '// &
                            'cuFFT plans failed')
      return
   endif
   ierr = cufftSetStream(self%plan_sd2z, acc_get_cuda_stream(acc_async_sync))
   ierr = ierr + cufftSetStream(self%plan_sz2d, acc_get_cuda_stream(acc_async_sync))
   ierr = ierr + cufftSetStream(self%plan_sz2z, acc_get_cuda_stream(acc_async_sync))
   if (ierr /= CUFFT_SUCCESS) then
      call elisha_set_error(stat, errmsg, ELISHA_ERR_FFT_PLAN, 'elisha_spectral_solver_cufft_object%slab_make_plans: '// &
                            'cufftSetStream failed')
      return
   endif
   self%has_slab_plans = .true.
   me = self%comm%rank
   allocate(self%yown(0:self%m(2)-1))
   do r=0, self%comm%nranks-1
      if (self%yr(r+1) > self%yr(r)) self%yown(self%yr(r):self%yr(r+1)-1) = r
   enddo
   nb = max(self%h1 * self%m(2) * (self%zr(me+1) - self%zr(me)), self%h1 * (self%yr(me+1) - self%yr(me)) * self%m(3), 1)
   allocate(self%t_sb(nb), self%t_rb(nb))
   !$acc enter data create(self%slab_r, self%slab_c, self%pen, self%t_sb, self%t_rb)
   self%slab_on_device = .true.
   ! the setup-time constants of the kernels, resident (M7, audit C5)
   !$acc enter data copyin(self%yr, self%yown)
   self%maps_on_device = .true.
   call place_lam(self)
   endsubroutine slab_make_plans_cufft

   subroutine slab_destroy_plans_cufft(self)
   !< Destroy the cuFFT slab plans.
   class(elisha_spectral_solver_cufft_object), intent(inout) :: self !< The solver.
   integer(c_int)                                            :: ierr !< cuFFT status.

   if (self%has_slab_plans) then
      ierr = cufftDestroy(self%plan_sd2z)
      ierr = cufftDestroy(self%plan_sz2d)
      ierr = cufftDestroy(self%plan_sz2z)
   endif
   if (self%slab_on_device) then
      !$acc exit data delete(self%slab_r, self%slab_c, self%pen, self%t_sb, self%t_rb)
      if (allocated(self%khat_pen)) then
         !$acc exit data delete(self%khat_pen)
      endif
   endif
   if (self%maps_on_device) then
      !$acc exit data delete(self%yr, self%yown)
   endif
   self%maps_on_device = .false.
   ierr = 0_c_int
   self%has_slab_plans = .false.
   self%slab_on_device = .false.
   if (allocated(self%yown)) deallocate(self%yown)
   if (allocated(self%t_sb)) deallocate(self%t_sb)
   if (allocated(self%t_rb)) deallocate(self%t_rb)
   endsubroutine slab_destroy_plans_cufft

   subroutine slab_planes_cufft(self, forward, stat, errmsg)
   !< Every local plane on the GPU: forward slab_r -> slab_c (D2Z), backward slab_c -> slab_r (Z2D), on the device.
   class(elisha_spectral_solver_cufft_object), intent(inout)         :: self    !< The solver.
   logical,                                    intent(in)            :: forward !< Direction.
   integer(I4P),                               intent(out)           :: stat    !< Status.
   character(:), allocatable,                  intent(out), optional :: errmsg  !< Message.
   integer(I4P)                                                      :: nzl     !< Local planes.
   integer(c_int)                                                    :: ierr    !< cuFFT status.

   stat = ELISHA_OK
   nzl = self%zr(self%comm%rank+1) - self%zr(self%comm%rank)
   if (nzl == 0) return
   call planes_dev(plan=merge(self%plan_sd2z, self%plan_sz2d, forward), m1=self%m(1), m2=self%m(2), h1=self%h1, &
                   nz=nzl, forward=forward, r=self%slab_r, c=self%slab_c, ierr=ierr)
   if (ierr /= CUFFT_SUCCESS) call elisha_set_error(stat, errmsg, ELISHA_ERR_FFT_PLAN, &
                                                    'elisha_spectral_solver_cufft_object: slab plane transform failed')
   endsubroutine slab_planes_cufft

   subroutine slab_lines_cufft(self, forward, stat, errmsg)
   !< Every local z-line on the GPU, in place in pen (Z2Z, forward or inverse), on the device.
   class(elisha_spectral_solver_cufft_object), intent(inout)         :: self    !< The solver.
   logical,                                    intent(in)            :: forward !< Direction.
   integer(I4P),                               intent(out)           :: stat    !< Status.
   character(:), allocatable,                  intent(out), optional :: errmsg  !< Message.
   integer(I4P)                                                      :: nyl     !< Local rows.
   integer(c_int)                                                    :: ierr    !< cuFFT status.

   stat = ELISHA_OK
   nyl = self%yr(self%comm%rank+1) - self%yr(self%comm%rank)
   if (nyl == 0) return
   call lines_dev(plan=self%plan_sz2z, m3=self%m(3), h1=self%h1, ny=nyl, forward=forward, p=self%pen, ierr=ierr)
   if (ierr /= CUFFT_SUCCESS) call elisha_set_error(stat, errmsg, ELISHA_ERR_FFT_PLAN, &
                                                    'elisha_spectral_solver_cufft_object: slab line transform failed')
   endsubroutine slab_lines_cufft

   subroutine slab_transpose_cufft(self, forward, stat, errmsg)
   !< The transpose on the device: forward slab_c(x, y, z local) -> pen(z, x, y local), backward the reverse; the blocks
   !< packed by destination rank (the host layout), the exchange on the device buffers (skipped on one rank).
   class(elisha_spectral_solver_cufft_object), intent(inout)         :: self    !< The solver.
   logical,                                    intent(in)            :: forward !< Direction.
   integer(I4P),                               intent(out)           :: stat    !< Status.
   character(:), allocatable,                  intent(out), optional :: errmsg  !< Message.
   integer(I4P), allocatable :: sc(:), sd(:), rc(:), rd(:) !< Counts and displacements (values).
   integer(I4P)              :: nr, me, r, nzl, nyl, nb    !< Counters, sizes.
   integer(I4P)              :: code                       !< Status of the exchange.

   stat = ELISHA_OK
   nr = self%comm%nranks ; me = self%comm%rank
   nzl = self%zr(me+1) - self%zr(me) ; nyl = self%yr(me+1) - self%yr(me)
   nb = size(self%t_sb)
   if (forward) then
      if (nzl > 0) call pack_planes_dev(h1=self%h1, m2=self%m(2), nz=nzl, nr=nr, yr=self%yr, yown=self%yown, &
                                        c=self%slab_c, nb=nb, sb=self%t_sb)
   else
      if (nyl > 0) call pack_lines_dev(m3=self%m(3), h1=self%h1, ny=nyl, p=self%pen, nb=nb, sb=self%t_sb)
   endif
   if (nr > 1) then
      allocate(sc(0:nr-1), sd(0:nr-1), rc(0:nr-1), rd(0:nr-1))
      do r=0, nr-1
         if (forward) then
            sc(r) = self%h1 * (self%yr(r+1) - self%yr(r)) * nzl
            rc(r) = self%h1 * nyl * (self%zr(r+1) - self%zr(r))
         else
            sc(r) = self%h1 * nyl * (self%zr(r+1) - self%zr(r))
            rc(r) = self%h1 * (self%yr(r+1) - self%yr(r)) * nzl
         endif
      enddo
      sd(0) = 0 ; rd(0) = 0
      do r=1, nr-1
         sd(r) = sd(r-1) + sc(r-1) ; rd(r) = rd(r-1) + rc(r-1)
      enddo
      call alltoallv_dev(comm=self%comm, nb=nb, sb=self%t_sb, sc=sc, sd=sd, rb=self%t_rb, rc=rc, rd=rd, stat=code)
      if (code /= ELISHA_OK) then
         call elisha_set_error(stat, errmsg, code, 'elisha_spectral_solver_cufft_object%slab_transpose: a collective '// &
                               'failed')
         return
      endif
      if (forward) then
         if (nyl > 0) call unpack_lines_dev(m3=self%m(3), h1=self%h1, ny=nyl, nb=nb, rb=self%t_rb, p=self%pen)
      else
         if (nzl > 0) call unpack_planes_dev(h1=self%h1, m2=self%m(2), nz=nzl, nr=nr, yr=self%yr, yown=self%yown, &
                                             nb=nb, rb=self%t_rb, c=self%slab_c)
      endif
   else ! one rank: the packed buffer is the received one
      if (forward) then
         call unpack_lines_dev(m3=self%m(3), h1=self%h1, ny=nyl, nb=nb, rb=self%t_sb, p=self%pen)
      else
         call unpack_planes_dev(h1=self%h1, m2=self%m(2), nz=nzl, nr=nr, yr=self%yr, yown=self%yown, nb=nb, &
                                rb=self%t_sb, c=self%slab_c)
      endif
   endif
   endsubroutine slab_transpose_cufft

   subroutine slab_spectral_cufft(self, factor)
   !< The spectral solve on the device: the division by the eigenvalues (the zero modes dropped) or, isolated, the
   !< product with the kernel's spectrum.
   class(elisha_spectral_solver_cufft_object), intent(inout) :: self   !< The solver.
   real(R_P),                                  intent(in)    :: factor !< Source factor.
   integer(I4P)                                              :: nyl    !< Local rows.

   nyl = self%yr(self%comm%rank+1) - self%yr(self%comm%rank)
   if (nyl == 0) return
   if (self%isolated) then
      call convolve_lines_dev(m3=self%m(3), h1=self%h1, ny=nyl, factor=factor, k=self%khat_pen, p=self%pen)
   else
      call divide_lines_dev(m3=self%m(3), h1=self%h1, ny=nyl, y0=self%yr(self%comm%rank),                  &
                            n1=size(self%lam1), n2=size(self%lam2), n3=size(self%lam3),                       &
                            scale=factor / real(product(int(self%nt, I8P)), R_P),                             &
                            l1=self%lam1, l2=self%lam2, l3=self%lam3, p=self%pen)
   endif
   endsubroutine slab_spectral_cufft

   subroutine slab_keep_kernel_cufft(self)
   !< Isolated, at setup: `khat_pen <- real(pen) / prod(m)` on the device, where it stays.
   class(elisha_spectral_solver_cufft_object), intent(inout) :: self !< The solver.

   !$acc enter data create(self%khat_pen)
   call keep_kernel_dev(n=size(self%pen), d=real(product(int(self%m, I8P)), R_P), p=self%pen, k=self%khat_pen)
   endsubroutine slab_keep_kernel_cufft

   subroutine slab_update_cufft(self, to_device)
   !< `slab_r` to (`to_device`) or from the device.
   class(elisha_spectral_solver_cufft_object), intent(inout) :: self      !< The solver.
   logical,                                    intent(in)    :: to_device !< Direction.

   if (self%zr(self%comm%rank+1) == self%zr(self%comm%rank)) return
   if (to_device) then
      !$acc update device(self%slab_r)
   else
      !$acc update self(self%slab_r)
   endif
   endsubroutine slab_update_cufft

   ! private procedures
   subroutine place_lam(self)
   !< The eigenvalue parts to the device, once per setup (M7, audit C5): `make_plans` or `slab_make_plans` calls it,
   !< `destroy_plans` releases them.
   class(elisha_spectral_solver_cufft_object), intent(inout) :: self !< The solver.

   if (self%lam_on_device) return
   !$acc enter data copyin(self%lam1, self%lam2, self%lam3)
   self%lam_on_device = .true.
   endsubroutine place_lam

   subroutine planes_dev(plan, m1, m2, h1, nz, forward, r, c, ierr)
   !< The 2-D transform of every plane on the device (r and c present), one plan executed on each plane in turn.
   integer(c_int), intent(in)    :: plan            !< cuFFT plan: one plane, D2Z or Z2D.
   integer(I4P),   intent(in)    :: m1, m2, h1, nz  !< Plane sizes, planes.
   logical,        intent(in)    :: forward         !< D2Z (r -> c) or Z2D (c -> r).
   real(R_P),      intent(inout) :: r(m1,m2,nz)     !< Real planes.
   complex(R_P),   intent(inout) :: c(h1,m2,nz)     !< Their transforms.
   integer(c_int), intent(out)   :: ierr            !< cuFFT status (ADAM stopped on a failure).
   integer(I4P)                  :: k               !< Counter.

   ierr = CUFFT_SUCCESS
   if (forward) then
      !$acc host_data use_device(r, c)
      do k=1, nz
         ierr = ierr + cufftExecD2Z(plan, r(1,1,k), c(1,1,k))
      enddo
      !$acc end host_data
   else
      !$acc host_data use_device(c, r)
      do k=1, nz
         ierr = ierr + cufftExecZ2D(plan, c(1,1,k), r(1,1,k))
      enddo
      !$acc end host_data
   endif
   endsubroutine planes_dev

   subroutine lines_dev(plan, m3, h1, ny, forward, p, ierr)
   !< The 1-D transform of every z-line on the device (p present), in place, one plan (a row of h1 lines) on each row
   !< in turn.
   integer(c_int), intent(in)    :: plan         !< cuFFT plan: one row of lines, Z2Z.
   integer(I4P),   intent(in)    :: m3, h1, ny   !< Line length, lines per row, rows.
   logical,        intent(in)    :: forward      !< CUFFT_FORWARD or CUFFT_INVERSE.
   complex(R_P),   intent(inout) :: p(m3,h1,ny)  !< Lines.
   integer(c_int), intent(out)   :: ierr         !< cuFFT status (ADAM stopped on a failure).
   integer(c_int)                :: dir          !< Direction.
   integer(I4P)                  :: j            !< Counter.

   ierr = CUFFT_SUCCESS
   dir = merge(CUFFT_FORWARD, CUFFT_INVERSE, forward)
   !$acc host_data use_device(p)
   do j=1, ny
      ierr = ierr + cufftExecZ2Z(plan, p(1,1,j), p(1,1,j), dir)
   enddo
   !$acc end host_data
   endsubroutine lines_dev

   subroutine pack_planes_dev(h1, m2, nz, nr, yr, yown, c, nb, sb)
   !< Forward transpose, on the device: the planes' rows by destination rank r (the rows [yr(r), yr(r+1)) of every
   !< local plane, x fastest, then y, then z), at r's displacement h1 nz yr(r).
   integer(I4P), intent(in)    :: h1, m2, nz, nr, nb !< Sizes, ranks, buffer size.
   integer(I4P), intent(in)    :: yr(0:nr)           !< First y-row of each rank.
   integer(I4P), intent(in)    :: yown(0:m2-1)       !< Owner rank of each y-row.
   complex(R_P), intent(in)    :: c(h1,m2,nz)        !< Transformed planes (present).
   complex(R_P), intent(inout) :: sb(nb)             !< Send buffer (present).
   integer(I4P)                :: x, y, z, r         !< Counters, rank.

   !$acc parallel loop collapse(3) present(c, sb, yr, yown) private(r)
   do z=1, nz
      do y=1, m2
         do x=1, h1
            r = yown(y-1)
            sb(h1*nz*yr(r) + x + h1*((y-1-yr(r)) + (yr(r+1)-yr(r))*(z-1))) = c(x,y,z)
         enddo
      enddo
   enddo
   endsubroutine pack_planes_dev

   subroutine unpack_planes_dev(h1, m2, nz, nr, yr, yown, nb, rb, c)
   !< Backward transpose, on the device: the planes' rows from each source rank (the inverse of pack_planes_dev).
   integer(I4P), intent(in)    :: h1, m2, nz, nr, nb !< Sizes, ranks, buffer size.
   integer(I4P), intent(in)    :: yr(0:nr)           !< First y-row of each rank.
   integer(I4P), intent(in)    :: yown(0:m2-1)       !< Owner rank of each y-row.
   complex(R_P), intent(in)    :: rb(nb)             !< Receive buffer (present).
   complex(R_P), intent(inout) :: c(h1,m2,nz)        !< Transformed planes (present).
   integer(I4P)                :: x, y, z, r         !< Counters, rank.

   !$acc parallel loop collapse(3) present(c, rb, yr, yown) private(r)
   do z=1, nz
      do y=1, m2
         do x=1, h1
            r = yown(y-1)
            c(x,y,z) = rb(h1*nz*yr(r) + x + h1*((y-1-yr(r)) + (yr(r+1)-yr(r))*(z-1)))
         enddo
      enddo
   enddo
   endsubroutine unpack_planes_dev

   subroutine pack_lines_dev(m3, h1, ny, p, nb, sb)
   !< Backward transpose, on the device: the z-lines as (x, y local, z), so the planes of rank r start at h1 ny zr(r).
   integer(I4P), intent(in)    :: m3, h1, ny, nb !< Sizes, buffer size.
   complex(R_P), intent(in)    :: p(m3,h1,ny)    !< Lines (present).
   complex(R_P), intent(inout) :: sb(nb)         !< Send buffer (present).
   integer(I4P)                :: x, y, z        !< Counters.

   !$acc parallel loop collapse(3) present(p, sb)
   do y=1, ny
      do x=1, h1
         do z=1, m3
            sb(x + h1*(y-1) + h1*ny*(z-1)) = p(z,x,y)
         enddo
      enddo
   enddo
   endsubroutine pack_lines_dev

   subroutine unpack_lines_dev(m3, h1, ny, nb, rb, p)
   !< Forward transpose, on the device: the z-lines from (x, y local, z) (the inverse of pack_lines_dev).
   integer(I4P), intent(in)    :: m3, h1, ny, nb !< Sizes, buffer size.
   complex(R_P), intent(in)    :: rb(nb)         !< Receive buffer (present).
   complex(R_P), intent(inout) :: p(m3,h1,ny)    !< Lines (present).
   integer(I4P)                :: x, y, z        !< Counters.

   !$acc parallel loop collapse(3) present(p, rb)
   do y=1, ny
      do x=1, h1
         do z=1, m3
            p(z,x,y) = rb(x + h1*(y-1) + h1*ny*(z-1))
         enddo
      enddo
   enddo
   endsubroutine unpack_lines_dev

   subroutine alltoallv_dev(comm, nb, sb, sc, sd, rb, rc, rd, stat)
   !< The exchange of the device buffers (GPU-aware MPI): `elisha_comm_object%alltoallv_c` inside `host_data`, so its
   !< assumed-size dummies receive the device addresses and pass them to `MPI_ALLTOALLV` (ADAM called `MPI_ALLTOALLV`
   !< here on `MPI_COMM_WORLD`).
   type(elisha_comm_object), intent(in)    :: comm                           !< Communication context.
   integer(I4P),             intent(in)    :: nb                             !< Buffer size.
   complex(R_P),             intent(in)    :: sb(nb)                         !< Send buffer (present).
   integer(I4P),             intent(in)    :: sc(0:), sd(0:), rc(0:), rd(0:) !< Counts and displacements (values).
   complex(R_P),             intent(inout) :: rb(nb)                         !< Receive buffer (present).
   integer(I4P),             intent(out)   :: stat                           !< Status of the exchange.

   !$acc host_data use_device(sb, rb)
   call comm%alltoallv_c(sb, sc, sd, rb, rc, rd, stat)
   !$acc end host_data
   endsubroutine alltoallv_dev

   subroutine divide_lines_dev(m3, h1, ny, y0, n1, n2, n3, scale, l1, l2, l3, p)
   !< The periodic spectral solve on the device: p <- p scale / lambda, the zero modes dropped (the host slab_spectral).
   integer(I4P), intent(in)    :: m3, h1, ny, y0 !< Sizes, first local row.
   integer(I4P), intent(in)    :: n1, n2, n3     !< Eigenvalue parts' sizes.
   real(R_P),    intent(in)    :: scale          !< Factor over the unnormalised transform pair.
   real(R_P),    intent(in)    :: l1(0:n1-1)     !< Eigenvalue part along x (present).
   real(R_P),    intent(in)    :: l2(0:n2-1)     !< Eigenvalue part along y (present).
   real(R_P),    intent(in)    :: l3(0:n3-1)     !< Eigenvalue part along z (present).
   complex(R_P), intent(inout) :: p(m3,h1,ny)    !< Lines (present).
   real(R_P)                   :: lam            !< Eigenvalue of the mode.
   integer(I4P)                :: i, j, k        !< Counters.

   !$acc parallel loop collapse(3) present(p, l1, l2, l3) private(lam)
   do j=1, ny
      do i=1, h1
         do k=1, m3
            lam = l1(i-1) + l2(y0+j-1) + l3(k-1)
            if (lam == 0._R_P) then
               p(k,i,j) = (0._R_P, 0._R_P)
            else
               p(k,i,j) = p(k,i,j) * (scale / lam)
            endif
         enddo
      enddo
   enddo
   endsubroutine divide_lines_dev

   subroutine convolve_lines_dev(m3, h1, ny, factor, k, p)
   !< The isolated spectral solve on the device: p <- p factor khat_pen.
   integer(I4P), intent(in)    :: m3, h1, ny  !< Sizes.
   real(R_P),    intent(in)    :: factor      !< Source factor.
   real(R_P),    intent(in)    :: k(m3,h1,ny) !< Kernel's spectrum over prod(m) (present).
   complex(R_P), intent(inout) :: p(m3,h1,ny) !< Lines (present).
   integer(I4P)                :: i, j, z     !< Counters.

   !$acc parallel loop collapse(3) present(p, k)
   do j=1, ny
      do i=1, h1
         do z=1, m3
            p(z,i,j) = p(z,i,j) * (factor * k(z,i,j))
         enddo
      enddo
   enddo
   endsubroutine convolve_lines_dev

   subroutine keep_kernel_dev(n, d, p, k)
   !< k <- real(p) / d on the device (the kernel's spectrum, real: the kernel is even).
   integer(I4P), intent(in)    :: n    !< Values.
   real(R_P),    intent(in)    :: d    !< Divisor, prod(m).
   complex(R_P), intent(in)    :: p(n) !< Transformed kernel (present).
   real(R_P),    intent(inout) :: k(n) !< Its spectrum (present).
   integer(I4P)                :: i    !< Counter.

   !$acc parallel loop present(p, k)
   do i=1, n
      k(i) = real(p(i), R_P) / d
   enddo
   endsubroutine keep_kernel_dev

   subroutine spectral_solve_dev(plan_d2z, plan_z2d, n1, n2, n3, scale, lam1, lam2, lam3, f, f_hat, ierr)
   !< The spectral solve on the device: D2Z, division by the eigenvalues, Z2D; f in, the potential out (in place).
   integer(c_int), intent(in)    :: plan_d2z            !< cuFFT forward plan.
   integer(c_int), intent(in)    :: plan_z2d            !< cuFFT backward plan.
   integer(I4P),   intent(in)    :: n1, n2, n3          !< Global cells per direction.
   real(R_P),      intent(in)    :: scale               !< Factor over the unnormalised transform pair.
   real(R_P),      intent(in)    :: lam1(0:n1/2)        !< Eigenvalue part along x.
   real(R_P),      intent(in)    :: lam2(0:n2-1)        !< Eigenvalue part along y.
   real(R_P),      intent(in)    :: lam3(0:n3-1)        !< Eigenvalue part along z.
   real(R_P),      intent(inout) :: f(n1,n2,n3)         !< Source in, potential out.
   complex(R_P),   intent(inout) :: f_hat(n1/2+1,n2,n3) !< Transform workspace.
   integer(c_int), intent(out)   :: ierr                !< cuFFT status, the forward one's if it failed.
   integer(c_int)                :: ierr_d2z            !< cuFFT status of the forward transform.
   real(R_P)                     :: lam                 !< Eigenvalue of the mode.
   integer(I4P)                  :: i, j, k             !< Counters.

   !$acc data copy(f) create(f_hat)
   !$acc host_data use_device(f, f_hat)
   ierr = cufftExecD2Z(plan_d2z, f, f_hat)
   !$acc end host_data
   ierr_d2z = ierr ! ADAM stopped here on a failure
   !$acc parallel loop collapse(3) present(f_hat, lam1, lam2, lam3) private(lam)
   do k=1, n3
      do j=1, n2
         do i=1, n1/2+1
            lam = lam1(i-1) + lam2(j-1) + lam3(k-1)
            if (lam == 0._R_P) then
               f_hat(i,j,k) = (0._R_P, 0._R_P)
            else
               f_hat(i,j,k) = f_hat(i,j,k) * (scale / lam)
            endif
         enddo
      enddo
   enddo
   !$acc host_data use_device(f_hat, f)
   ierr = cufftExecZ2D(plan_z2d, f_hat, f)
   !$acc end host_data
   !$acc end data
   if (ierr_d2z /= CUFFT_SUCCESS) ierr = ierr_d2z
   endsubroutine spectral_solve_dev

   subroutine spectral_convolve_dev(plan_d2z, plan_z2d, m1, m2, m3, factor, khat, f, f_hat, ierr)
   !< The isolated solve on the device: D2Z, the product with the kernel's spectrum (already on the device), Z2D; the
   !< padded source in, the potential out (in place).
   integer(c_int), intent(in)    :: plan_d2z            !< cuFFT forward plan.
   integer(c_int), intent(in)    :: plan_z2d            !< cuFFT backward plan.
   integer(I4P),   intent(in)    :: m1, m2, m3          !< Transform cells per direction.
   real(R_P),      intent(in)    :: factor              !< Source factor.
   real(R_P),      intent(in)    :: khat(m1/2+1,m2,m3)  !< Kernel's spectrum over prod(m), on the device.
   real(R_P),      intent(inout) :: f(m1,m2,m3)         !< Padded source in, potential out.
   complex(R_P),   intent(inout) :: f_hat(m1/2+1,m2,m3) !< Transform workspace.
   integer(c_int), intent(out)   :: ierr                !< cuFFT status, the forward one's if it failed.
   integer(c_int)                :: ierr_d2z            !< cuFFT status of the forward transform.
   integer(I4P)                  :: i, j, k             !< Counters.

   !$acc data copy(f) create(f_hat)
   !$acc host_data use_device(f, f_hat)
   ierr = cufftExecD2Z(plan_d2z, f, f_hat)
   !$acc end host_data
   ierr_d2z = ierr ! ADAM stopped here on a failure
   !$acc parallel loop collapse(3) present(f_hat, khat)
   do k=1, m3
      do j=1, m2
         do i=1, m1/2+1
            f_hat(i,j,k) = f_hat(i,j,k) * (factor * khat(i,j,k))
         enddo
      enddo
   enddo
   !$acc host_data use_device(f_hat, f)
   ierr = cufftExecZ2D(plan_z2d, f_hat, f)
   !$acc end host_data
   !$acc end data
   if (ierr_d2z /= CUFFT_SUCCESS) ierr = ierr_d2z
   endsubroutine spectral_convolve_dev

   subroutine kernel_spectrum_dev(plan_d2z, m1, m2, m3, f, f_hat, ierr)
   !< The kernel's forward transform on the device: f in, its spectrum out to the host.
   integer(c_int), intent(in)    :: plan_d2z            !< cuFFT forward plan.
   integer(I4P),   intent(in)    :: m1, m2, m3          !< Transform cells per direction.
   real(R_P),      intent(in)    :: f(m1,m2,m3)         !< Kernel.
   complex(R_P),   intent(inout) :: f_hat(m1/2+1,m2,m3) !< Its spectrum.
   integer(c_int), intent(out)   :: ierr                !< cuFFT status.

   !$acc data copyin(f) copyout(f_hat)
   !$acc host_data use_device(f, f_hat)
   ierr = cufftExecD2Z(plan_d2z, f, f_hat)
   !$acc end host_data
   !$acc end data
   endsubroutine kernel_spectrum_dev
endmodule elisha_spectral_solver_cufft
