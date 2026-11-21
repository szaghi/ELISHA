! the 7-point Laplacian of phi, read through the ghost layers the solver filled, against f (mean f is 0); and phi
! against the continuum solution -f / (12 pi^2)
res = 0._R_P ; err = 0._R_P ; n = 0
do b=1, mesh%nb ; do k=1, nc ; do j=1, nc ; do i=1, nc
   n = n + 1
   lap = (phi(i+1,j,k,b) + phi(i-1,j,k,b) + phi(i,j+1,k,b) + phi(i,j-1,k,b) + phi(i,j,k+1,b) + phi(i,j,k-1,b) &
          - 6._R_P * phi(i,j,k,b)) / h**2
   res = max(res, abs(lap - f(n)))
   err = max(err, abs(phi(i,j,k,b) + f(n) / (12._R_P * PI**2)))
enddo ; enddo ; enddo ; enddo
res = res / maxval(abs(f)) ; err = err * 12._R_P * PI**2 / maxval(abs(f))
