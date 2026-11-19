!< ELISHA test: the centred D1 rows of the wide operator D(G) equal GIDEON's, bitwise (a test-only dependency).
program elisha_test_gideon_d1
!< ELISHA test: the centred D1 rows of the wide operator D(G) equal GIDEON's, bitwise (a test-only dependency).
!<
!< `ELISHA_D1_COEF(:,p/2)` (the a_j of `elisha_spectral`, ADAM's `D1_COEF`) against GIDEON's generated centred
!< first-derivative rows `FD_D1_S1..S4` (`gideon_tables`, exact rationals by GIDEON's generator): two independent
!< sources of one table. ELISHA keeps its own copy (the libraries stay separate); this test keeps them equal.
use, intrinsic :: iso_fortran_env, only : int64
use elisha, only : I4P, ELISHA_D1_COEF
use gideon_tables, only : FD_D1_S1, FD_D1_S2, FD_D1_S3, FD_D1_S4
implicit none
integer(I4P) :: failures !< Failed checks.

failures = 0
call compare(1, FD_D1_S1)
call compare(2, FD_D1_S2)
call compare(3, FD_D1_S3)
call compare(4, FD_D1_S4)
if (any(ELISHA_D1_COEF(2:4,1) /= 0) .or. any(ELISHA_D1_COEF(3:4,2) /= 0) .or. ELISHA_D1_COEF(4,3) /= 0) then
   failures = failures + 1
   print '(A)', '  FAIL nonzero D1 entries beyond the stencil reach'
endif
if (failures > 0) error stop 1
print '(A)', 'elisha_test_gideon_d1: ok (orders 2, 4, 6, 8 bitwise GIDEON)'
contains
   subroutine compare(h, row)
   !< Column h of ELISHA_D1_COEF against GIDEON's row of order 2h, by bit pattern.
   integer(I4P), intent(in) :: h      !< Half order.
   real(kind(ELISHA_D1_COEF)), intent(in) :: row(:) !< GIDEON's row.

   if (size(row) /= h) then
      failures = failures + 1
      print '(A,I0)', '  FAIL GIDEON row size, order ', 2 * h
   elseif (any(transfer(ELISHA_D1_COEF(1:h,h), 1_int64, h) /= transfer(row, 1_int64, h))) then
      failures = failures + 1
      print '(A,I0)', '  FAIL D1 not bitwise GIDEON, order ', 2 * h
   endif
   endsubroutine compare
endprogram elisha_test_gideon_d1
