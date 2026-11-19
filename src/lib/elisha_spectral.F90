!< ELISHA, L0 spectral math: the axis eigenvalues, the wide operator's stencil and the transform sizes.
module elisha_spectral
!< ELISHA, L0 spectral math: the axis eigenvalues, the wide operator's stencil and the transform sizes.
!<
!< Pure functions of integers and spacings, no state, no MPI, no FFT library: what the spectral solver computes once
!< per setup. Ported from ADAM's `adam_poisson_fft_object` (ADAM 669d0bdd) bitwise: the same expressions in the same
!< order (`src/tests/elisha_test_adam_parity_l0` holds them to it).
!<
!< - `elisha_axis_eigenvalue(k, nt, dx, p)`: mode k of an axis on the nt-cell periodic transform. The compact
!<   3-point operator, `(2 cos(theta) - 2) / dx^2`, or the wide D(G) of order p (the centred first derivative of
!<   order p composed with itself), `-s_p(theta)^2 / dx^2` with `s_p(theta) = sum_j 2 a_j sin(j theta)`, set exactly
!<   to 0 at theta = 0 and pi (`sin(pi)` is 1e-16, not 0: the checkerboard is in the kernel of every centred D1).
!< - `elisha_dg_stencil(p)`: the 1-D stencil of D1 D1 times h^2, offsets -p..p.
!< - `elisha_smooth_size(k)`: the smallest even size at or above k with no prime factor above 13 (FFTW has
!<   hard-coded transforms for these factors; ADAM #84 E5b measured 3.3x on 262 = 2 x 131 against 264).
!< - `elisha_axis_transform_size(n, ngc, bc, isolated, pcg, null)`: the transform cells of an axis of n cells: n
!<   (periodic, a null axis, the CG path), 2 (n + 1) (ghost-Dirichlet: odd about the ghost centres), 2n or 4n
!<   (Dirichlet/Neumann faces: odd/even about each face, 4n when they differ), or the smooth size at or above
!<   2 (n + ngc) (isolated: zero padding).
use penf, only : I4P
use elisha_kind_parameters, only : R_P
use elisha_parameters, only : ELISHA_BC_PERIODIC, ELISHA_BC_DIRICHLET_GHOST
implicit none
private
public :: ELISHA_D1_COEF
public :: elisha_axis_eigenvalue
public :: elisha_axis_transform_size
public :: elisha_dg_stencil
public :: elisha_smooth_size

real(R_P), parameter :: PI=acos(-1._R_P) !< Pi.
! centred first-derivative coefficients a_j of order p: D1 q_i = sum_j a_j (q_{i+j} - q_{i-j}) / h (column p/2)
real(R_P), parameter :: ELISHA_D1_COEF(4,4)=reshape([1._R_P/2._R_P, 0._R_P, 0._R_P, 0._R_P,                         &
                                                     2._R_P/3._R_P, -1._R_P/12._R_P, 0._R_P, 0._R_P,                 &
                                                     3._R_P/4._R_P, -3._R_P/20._R_P, 1._R_P/60._R_P, 0._R_P,         &
                                                     4._R_P/5._R_P, -1._R_P/5._R_P, 4._R_P/105._R_P, -1._R_P/280._R_P], &
                                                    [4, 4]) !< a_j [j, p/2].

contains
   pure function elisha_smooth_size(k) result(m)
   !< The smallest even size `m >= k` with no prime factor above 13.
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
   endfunction elisha_smooth_size

   pure function elisha_axis_eigenvalue(k, nt, dx, dg_order) result(lam)
   !< The eigenvalue part of mode k of an axis on the nt-cell periodic transform: the compact operator,
   !< `(2 cos(theta) - 2) / dx^2`, or the wide D(G) of order p, `-s_p(theta)^2 / dx^2`, exactly 0 at theta = 0 and pi.
   integer(I4P), intent(in) :: k, nt    !< Mode, transform cells.
   real(R_P),    intent(in) :: dx       !< Spacing.
   integer(I4P), intent(in) :: dg_order !< 0 compact, else the order of D(G).
   real(R_P)                :: lam      !< Eigenvalue part.
   real(R_P)                :: th, s    !< Phase, symbol of dx D1.
   integer(I4P)             :: j        !< Counter.

   if (dg_order == 0_I4P) then
      lam = (2._R_P * cos(2._R_P * PI * k / nt) - 2._R_P) / dx**2
      return
   endif
   lam = 0._R_P
   if (k == 0_I4P .or. 2_I4P * k == nt) return ! theta = 0 or pi: s_p vanishes exactly
   th = 2._R_P * PI * k / nt
   s = 0._R_P
   do j=1, dg_order/2
      s = s + 2._R_P * ELISHA_D1_COEF(j,dg_order/2) * sin(j * th)
   enddo
   lam = -(s * s) / dx**2
   endfunction elisha_axis_eigenvalue

   pure function elisha_dg_stencil(p) result(c)
   !< The 1-D stencil of D1 D1 (times h^2), offsets -p..p, the centred D1 of order p composed with itself.
   integer(I4P), intent(in) :: p          !< Order.
   real(R_P)                :: c(-8:8)    !< Stencil.
   real(R_P)                :: a(-4:4)    !< D1 (times h).
   integer(I4P)             :: j, o1, o2  !< Counters.

   a = 0._R_P
   do j=1, p/2
      a(j) = ELISHA_D1_COEF(j,p/2) ; a(-j) = -ELISHA_D1_COEF(j,p/2)
   enddo
   c = 0._R_P
   do o1=-p/2, p/2
      do o2=-p/2, p/2
         c(o1+o2) = c(o1+o2) + a(o1) * a(o2)
      enddo
   enddo
   endfunction elisha_dg_stencil

   pure function elisha_axis_transform_size(n, ngc, bc, isolated, pcg, null) result(m)
   !< The transform cells of an axis of n cells with the face kinds `bc` (see the module): ADAM's sizing, one axis.
   integer(I4P), intent(in) :: n        !< Cells along the axis.
   integer(I4P), intent(in) :: ngc      !< Ghost layers.
   integer(I4P), intent(in) :: bc(2)    !< Kinds of the low and high faces.
   logical,      intent(in) :: isolated !< Isolated boundaries (zero padding).
   logical,      intent(in) :: pcg      !< The CG path (zero-ghost or free-ghost faces with the wide D(G)).
   logical,      intent(in) :: null     !< A null axis.
   integer(I4P)             :: m        !< Transform cells.

   m = n
   if (pcg) return
   if (isolated) then
      m = elisha_smooth_size(2 * (n + ngc))
      return
   endif
   if (null) return
   if (all(bc == ELISHA_BC_DIRICHLET_GHOST)) then
      m = 2 * (n + 1) ! odd about the ghost centres 0 and n + 1
   elseif (all(bc /= ELISHA_BC_PERIODIC)) then
      m = merge(2, 4, bc(1) == bc(2)) * n ! odd/even about each face
   endif
   endfunction elisha_axis_transform_size
endmodule elisha_spectral
