! the analytic potential of the Gaussian is -M erf(r / (sqrt(2) sigma)) / r; the max error over the cells, relative to
! the potential at the centre
err = 0._R_P
do run=1, 3
   do b=1, mesh%nb ; do k=1, NC ; do j=1, NC ; do i=1, NC
      x = (mesh%coords(1:3,b) * NC + [i, j, k] - 0.5_R_P) * H
      err(run) = max(err(run), abs(phi(i,j,k,b,run) - potential(norm2(x - C))))
   enddo ; enddo ; enddo ; enddo
enddo
err = err / abs(potential(0._R_P))
print '(A,ES9.2)', 'isolated, spectral path:     max error ', err(1)
print '(A,ES9.2)', 'isolated, hierarchical path: max error ', err(2)
print '(A,ES9.2)', 'periodic:                    max error ', err(3)
print '(A,L1)', 'the two isolated paths agree to 1e-10: ', &
                maxval(abs(phi(1:NC,1:NC,1:NC,:,1) - phi(1:NC,1:NC,1:NC,:,2))) <= 1.e-10_R_P * abs(potential(0._R_P))
