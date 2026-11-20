subroutine report
!< The status: its code, its name and meaning (elisha_status), and the message of this failure.
if (stat == ELISHA_OK) then
   print '(A)', 'ok'
else
   print '(A,I0,A)', 'stat ', stat, ': '//elisha_status_name(stat)//' ('//elisha_status_text(stat)//')'
   print '(A)', '  '//errmsg
endif
endsubroutine report
