!< Test oracle: ADAM's L0 spectral math, verbatim from `adam_poisson_fft_object` (ADAM 669d0bdd).
module adam_flail_l0_reference
!< Test oracle: ADAM's L0 spectral math, verbatim from `adam_poisson_fft_object` (ADAM 669d0bdd).
!<
!< The bodies are ADAM's, copied line for line; ELISHA's `elisha_spectral` and `elisha_extension` must equal them
!< bitwise (`elisha_test_adam_parity_l0`). Two adaptations, both outside the arithmetic:
!< - `axis_map` was internal to `slab_setup` and read the solver's state (`self%n(d_)`, `self%m(d_)`, `self%bc(:,d_)`,
!<   `self%isolated`): here those are the arguments `n_`, `m_`, `bc_`, `isolated_`;
!< - `transform_sizes` is the sizing block of `initialize_blocks` (the `self%m` loop), with its inputs as arguments.
!< Not part of the library: never used by `src/lib`.
use :: adam_flail_parameters_reference, only : FLAIL_BC_PERIODIC, FLAIL_BC_DIRICHLET, FLAIL_BC_DIRICHLET_GHOST
use :: penf, only : I4P, R8P
implicit none
private
public :: D1_COEF, axis_eigenvalue, axis_map, dg_stencil, set_plane, smooth_size, transform_sizes

real(R8P), parameter :: PI=acos(-1._R8P) !< Pi.
! centred first-derivative coefficients a_j of order p: D1 q_i = sum_j a_j (q_{i+j} - q_{i-j}) / h (column p/2)
real(R8P), parameter :: D1_COEF(4,4)=reshape([1._R8P/2._R8P, 0._R8P, 0._R8P, 0._R8P,                         &
                                              2._R8P/3._R8P, -1._R8P/12._R8P, 0._R8P, 0._R8P,                 &
                                              3._R8P/4._R8P, -3._R8P/20._R8P, 1._R8P/60._R8P, 0._R8P,         &
                                              4._R8P/5._R8P, -1._R8P/5._R8P, 4._R8P/105._R8P, -1._R8P/280._R8P], &
                                             [4, 4]) !< a_j [j, p/2].


contains
   pure subroutine set_plane(f, d_, t, src, sg)
   !< The plane t of axis d_ set to sg times the plane src (cell indices; sg = 0 zeroes it).
   real(R8P),    intent(inout) :: f(1:,1:,1:) !< The array.
   integer(I4P), intent(in)    :: d_, t, src  !< Axis, target and source cell indices.
   real(R8P),    intent(in)    :: sg          !< Sign (0: zero).

   select case(d_)
   case(1)
      if (sg == 0._R8P) then
         f(t+1,:,:) = 0._R8P
      else
         f(t+1,:,:) = sg * f(src+1,:,:)
      endif
   case(2)
      if (sg == 0._R8P) then
         f(:,t+1,:) = 0._R8P
      else
         f(:,t+1,:) = sg * f(:,src+1,:)
      endif
   case(3)
      if (sg == 0._R8P) then
         f(:,:,t+1) = 0._R8P
      else
         f(:,:,t+1) = sg * f(:,:,src+1)
      endif
   endselect
   endsubroutine set_plane


   pure function smooth_size(k) result(m)
   !< The smallest even size `m >= k` with no prime factor above 13: FFTW has hard-coded transforms for these factors
   !< (#84 E5b, medooza: a 7-smooth rule moved the James step's 260 = 2^2 5 13 to 270 and cost it 15%; cuFFT took the
   !< same time on 262 and 270).
   integer(I4P), intent(in) :: k !< Least size.
   integer(I4P)             :: m !< Size.
   integer(I4P), parameter  :: PRIMES(6)=[2_I4P, 3_I4P, 5_I4P, 7_I4P, 11_I4P, 13_I4P] !< Allowed factors.
   integer(I4P)             :: r !< Remainder after the allowed factors.
   integer(I4P)             :: p !< Counter.

   m = k + mod(k, 2_I4P)
   do
      r = m
      do p=1, size(PRIMES)
         do while (mod(r, PRIMES(p)) == 0)
            r = r / PRIMES(p)
         enddo
      enddo
      if (r == 1) return
      m = m + 2_I4P
   enddo
   endfunction smooth_size

   pure function axis_eigenvalue(k, nt, dx, dg_order) result(lam)
   !< The eigenvalue part of mode k of an axis on the nt-cell periodic transform: the compact operator,
   !< `(2 cos(theta) - 2) / dx^2`, or the wide D(G) of order p, `-s_p(theta)^2 / dx^2`, exactly 0 at theta = 0 and pi.
   integer(I4P), intent(in) :: k, nt    !< Mode, transform cells.
   real(R8P),    intent(in) :: dx       !< Spacing.
   integer(I4P), intent(in) :: dg_order !< 0 compact, else the order of D(G).
   real(R8P)                :: lam      !< Eigenvalue part.
   real(R8P)                :: th, s    !< Phase, symbol of dx D1.
   integer(I4P)             :: j        !< Counter.

   if (dg_order == 0_I4P) then
      lam = (2._R8P * cos(2._R8P * PI * k / nt) - 2._R8P) / dx**2
      return
   endif
   lam = 0._R8P
   if (k == 0_I4P .or. 2_I4P * k == nt) return ! theta = 0 or pi: s_p vanishes exactly
   th = 2._R8P * PI * k / nt
   s = 0._R8P
   do j=1, dg_order/2
      s = s + 2._R8P * D1_COEF(j,dg_order/2) * sin(j * th)
   enddo
   lam = -(s * s) / dx**2
   endfunction axis_eigenvalue

   pure function dg_stencil(p) result(c)
   !< The 1-D stencil of D1 D1 (times h^2), offsets -p..p, the centred D1 of order p (`D1_COEF`) composed with itself.
   integer(I4P), intent(in) :: p          !< Order.
   real(R8P)                :: c(-8:8)    !< Stencil.
   real(R8P)                :: a(-4:4)    !< D1 (times h).
   integer(I4P)             :: j, o1, o2  !< Counters.

   a = 0._R8P
   do j=1, p/2
      a(j) = D1_COEF(j,p/2) ; a(-j) = -D1_COEF(j,p/2)
   enddo
   c = 0._R8P
   do o1=-p/2, p/2
      do o2=-p/2, p/2
         c(o1+o2) = c(o1+o2) + a(o1) * a(o2)
      enddo
   enddo
   endfunction dg_stencil

   recursive subroutine axis_map(n_, m_, bc_, isolated_, t_, src, sg)
   !< Target cell t_ (0-based) of an axis: its source cell (0-based, -1 for zero) and sign, as `extend_global`.
   integer(I4P), intent(in)  :: n_, m_, bc_(2) !< ADAM: self%n(d_), self%m(d_), self%bc(:,d_).
   logical,      intent(in)  :: isolated_      !< ADAM: self%isolated.
   integer(I4P), intent(in)  :: t_             !< Target cell.
   integer(I4P), intent(out) :: src            !< Source cell.
   real(R8P),    intent(out) :: sg             !< Sign.

   src = t_ ; sg = 1._R8P
   if (t_ < n_) return ! the domain itself
   src = -1 ; sg = 0._R8P
   if (isolated_) return ! the padding
   if (all(bc_ == FLAIL_BC_DIRICHLET_GHOST)) then ! odd about the ghost centres n and m - 1
      if (t_ == n_ .or. t_ == m_ - 1) return
      src = 2 * n_ - t_ ; sg = -1._R8P
   elseif (t_ < 2 * n_) then ! the reflection about the high face
      src = 2 * n_ - 1 - t_ ; sg = merge(-1._R8P, 1._R8P, bc_(2) == FLAIL_BC_DIRICHLET)
   else ! the reflection of [0, 2n) about the low face
      call axis_map(n_, m_, bc_, isolated_, 4 * n_ - 1 - t_, src, sg)
      sg = sg * merge(-1._R8P, 1._R8P, bc_(1) == FLAIL_BC_DIRICHLET)
   endif
   endsubroutine axis_map

   pure function transform_sizes(n, ngc, bc, isolated, pcg, null_) result(m)
   !< The sizing block of `initialize_blocks` (self%n, self%ngc, self%bc, self%isolated, self%pcg, null_ as arguments).
   integer(I4P), intent(in) :: n(3), ngc, bc(2,3) !< Cells, ghosts, kinds.
   logical,      intent(in) :: isolated, pcg, null_(3) !< Isolated, CG path, null axes.
   integer(I4P)             :: m(3)            !< Transform cells.
   integer(I4P)             :: d               !< Counter.

   m = n
   if (pcg) then
      continue
   elseif (isolated) then
      do d=1, 3 ! at least 2 (n + ngc); the next 13-smooth size (#84 E5b: FFTW slows on large prime factors)
         m(d) = smooth_size(2 * (n(d) + ngc))
      enddo
   else
      do d=1, 3
         if (null_(d)) cycle
         if (all(bc(:,d) == FLAIL_BC_DIRICHLET_GHOST)) then
            m(d) = 2 * (n(d) + 1) ! odd about the ghost centres 0 and n + 1
         elseif (all(bc(:,d) /= FLAIL_BC_PERIODIC)) then
            m(d) = merge(2, 4, bc(1,d) == bc(2,d)) * n(d) ! odd/even about each face
         endif
      enddo
   endif
   endfunction transform_sizes
endmodule adam_flail_l0_reference
