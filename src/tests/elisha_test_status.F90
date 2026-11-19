!< ELISHA test: the status codes and their helpers.
program elisha_test_status
!< ELISHA test: the status codes and their helpers.
!<
!< Every code from ELISHA_OK to ELISHA_STATUS_LAST has a distinct name `ELISHA_*` and a meaning; an out-of-range code
!< has neither; `elisha_set_error` sets the status always and the message `<name>: <msg>` only when asked.
use elisha, only : I4P, ELISHA_OK, ELISHA_STATUS_LAST, ELISHA_ERR_MESH_BALANCE, elisha_set_error, elisha_status_name, &
                   elisha_status_text
implicit none
integer(I4P)              :: failures, c, d, stat !< Failed checks, codes, status.
character(:), allocatable :: msg, nm              !< Message, name.

failures = 0
if (ELISHA_OK /= 0) call fail('ELISHA_OK is not 0')
do c=0, ELISHA_STATUS_LAST
   nm = elisha_status_name(c)
   if (nm(1:7) /= 'ELISHA_' .or. nm == 'ELISHA_UNKNOWN_STATUS') call fail('code without a name')
   if (len(elisha_status_text(c)) < 5 .or. elisha_status_text(c) == 'an unknown status code') &
      call fail('code without a meaning')
   do d=0, c-1
      if (elisha_status_name(c) == elisha_status_name(d)) call fail('two codes with one name')
   enddo
enddo
if (elisha_status_name(ELISHA_STATUS_LAST + 1) /= 'ELISHA_UNKNOWN_STATUS') call fail('out-of-range name')
if (elisha_status_name(-1) /= 'ELISHA_UNKNOWN_STATUS') call fail('negative code name')
call elisha_set_error(stat, msg, ELISHA_ERR_MESH_BALANCE, 'leaf (1,2,3) at level 4 next to level 2')
if (stat /= ELISHA_ERR_MESH_BALANCE) call fail('set_error: status')
if (msg /= 'ELISHA_ERR_MESH_BALANCE: leaf (1,2,3) at level 4 next to level 2') call fail('set_error: message')
call elisha_set_error(stat=stat, code=ELISHA_ERR_MESH_BALANCE, msg='no message asked')
if (stat /= ELISHA_ERR_MESH_BALANCE) call fail('set_error without errmsg: status')
if (failures > 0) error stop 1
print '(A,I0,A)', 'elisha_test_status: ok (', ELISHA_STATUS_LAST + 1, ' codes)'
contains
   subroutine fail(what)
   !< Report a failed check.
   character(*), intent(in) :: what !< What failed.

   failures = failures + 1
   print '(A)', '  FAIL '//what
   endsubroutine fail
endprogram elisha_test_status
