! the 7-point Laplacian of phi, read through the ghost layers the solver filled, against factor f (mean f is 0);
! and phi against the continuum solution -factor f / (12 pi^2): second order in H
res = 0._R_P ; err = 0._R_P ; fmax = maxval(abs(f)) ; n = 0
do b=1, mesh%nb ; do k=1, NC ; do j=1, NC ; do i=1, NC
   n = n + 1
   lap = (phi(i+1,j,k,b) + phi(i-1,j,k,b) + phi(i,j+1,k,b) + phi(i,j-1,k,b) + phi(i,j,k+1,b) + phi(i,j,k-1,b) &
          - 6._R_P * phi(i,j,k,b)) / H**2
   res = max(res, abs(lap - FACTOR * f(n)))
   err = max(err, abs(phi(i,j,k,b) + FACTOR * f(n) / (12._R_P * PI**2)))
enddo ; enddo ; enddo ; enddo
print '(A,L1)', 'residual below 1e-12 of the source: ', res <= 1.e-12_R_P * FACTOR * fmax
print '(A,ES8.2)', 'max |phi - phi_exact| / max |phi_exact|: ', err / (FACTOR * fmax / (12._R_P * PI**2))
