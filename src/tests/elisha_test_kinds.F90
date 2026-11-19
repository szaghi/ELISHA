!< ELISHA test: the working kinds reach a client through `use elisha` (the PENF dependency is wired).
program elisha_test_kinds
!< ELISHA test: the working kinds reach a client through `use elisha` (the PENF dependency is wired).
use elisha, only : I4P, I8P, R8P, R_P
implicit none

if (storage_size(1_I4P) /= 32)  error stop 'I4P is not a 32-bit integer'
if (storage_size(1_I8P) /= 64)  error stop 'I8P is not a 64-bit integer'
if (storage_size(1._R8P) /= 64) error stop 'R8P is not a 64-bit real'
if (R_P /= R8P)                 error stop 'R_P is not R8P'
print '(A)', 'elisha_test_kinds: ok'
endprogram elisha_test_kinds
