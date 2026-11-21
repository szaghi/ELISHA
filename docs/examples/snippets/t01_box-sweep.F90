! the same problem on 2^3 blocks of nc^3 cells, nc = 4, 8, 16, 32: the cell spacing halves at every level
print '(A)', '   cells   residual < 1e-12 |f|   max error   order'
err_prev = 0._R_P
do level=0, 3
   nc = 4 * 2**level
   call solve_box(nc, res, err)
   if (level == 0) then
      print '(I5,A,L12,ES20.3)', 2*nc, '^3', res < 1.e-12_R_P, err
   else
      print '(I5,A,L12,ES20.3,F8.2)', 2*nc, '^3', res < 1.e-12_R_P, err, log(err_prev / err) / log(2._R_P)
   endif
   err_prev = err
enddo
