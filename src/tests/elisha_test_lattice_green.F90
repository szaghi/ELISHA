!< ELISHA test: the lattice Green's function by its meaning (L_1 G = delta, Watson's G(0)) and bitwise against ADAM.
program elisha_test_lattice_green
!< ELISHA test: the lattice Green's function by its meaning (L_1 G = delta, Watson's G(0)) and bitwise against ADAM.
!<
!< - the residual `sum_d [G(n + e_d) - 2 G(n) + G(n - e_d)] - delta_(n,0)` on [-63, 63]^3, across the faces |n_d| = 32
!<   where the table meets the expansion: at most 1e-12 (the generator's composite check: 1e-13);
!< - G(0) against Watson's closed form for the simple cubic lattice, -0.252731009858663 (mpmath, by the generator):
!<   relative 1e-12; G(1,0,0) - G(0) = 1/6 to 1e-14;
!< - the far field against -1 / (4 pi r): relative 1e-4 at r = 100 (G1 is O(r^-3) relative);
!< - bitwise against ADAM's `lattice_green` (oracle `adam_lattice_green_library`, verbatim) on [-70, 70]^3.
use, intrinsic :: iso_fortran_env, only : int64
use elisha, only : I4P, R_P, elisha_lattice_green
use adam_lattice_green_library, only : lattice_green
implicit none
real(R_P), parameter :: WATSON_G0=-0.252731009858663_R_P !< Watson's G(0), mpmath.
real(R_P), parameter :: PI=acos(-1._R_P)                 !< Pi.
integer(I4P)         :: failures                         !< Failed checks.
integer(I4P)         :: i, j, k, d, nbits                !< Counters.
integer(I4P)         :: e(3,3)                           !< Unit offsets.
real(R_P)            :: res, rmax, g0, r                 !< Residual, its max, G(0), radius.

failures = 0
e = reshape([1, 0, 0, 0, 1, 0, 0, 0, 1], [3, 3])
rmax = 0._R_P
do k=-63, 63
   do j=-63, 63
      do i=-63, 63
         g0 = elisha_lattice_green([i, j, k])
         res = -merge(1._R_P, 0._R_P, i == 0 .and. j == 0 .and. k == 0)
         do d=1, 3
            res = res + elisha_lattice_green([i, j, k] + e(:,d)) - 2._R_P * g0 + elisha_lattice_green([i, j, k] - e(:,d))
         enddo
         rmax = max(rmax, abs(res))
      enddo
   enddo
enddo
print '(A,ES10.3)', 'elisha_test_lattice_green: max |L_1 G - delta| on [-63,63]^3 = ', rmax
if (rmax > 1.e-12_R_P) call fail('residual above 1e-12')
g0 = elisha_lattice_green([0, 0, 0])
if (abs(g0 - WATSON_G0) > 1.e-12_R_P * abs(WATSON_G0)) call fail('G(0) against Watson')
if (abs(elisha_lattice_green([1, 0, 0]) - g0 - 1._R_P / 6._R_P) > 1.e-14_R_P) call fail('G(1,0,0) - G(0) /= 1/6')
r = 100._R_P
if (abs(elisha_lattice_green([100, 0, 0]) + 1._R_P / (4._R_P * PI * r)) > 1.e-4_R_P / (4._R_P * PI * r)) &
   call fail('far field against -1 / (4 pi r)')
nbits = 0
do k=-70, 70
   do j=-70, 70
      do i=-70, 70
         if (transfer(elisha_lattice_green([i, j, k]), 1_int64) /= transfer(lattice_green([i, j, k]), 1_int64)) &
            nbits = nbits + 1
      enddo
   enddo
enddo
if (nbits > 0) call fail('not bitwise ADAM')
if (failures > 0) error stop 1
print '(A)', 'elisha_test_lattice_green: ok (bitwise ADAM on [-70,70]^3)'
contains
   subroutine fail(msg)
   !< Report a failed check.
   character(*), intent(in) :: msg !< What failed.

   failures = failures + 1
   print '(A)', '  FAIL '//msg
   endsubroutine fail
endprogram elisha_test_lattice_green
