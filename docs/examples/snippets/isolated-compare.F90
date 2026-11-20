! along the line y = z = 15.5 H, from x = 24.5 H out to the last block's ghost layer i = 34 (outside); the analytic
! potential of the Gaussian is -M erf(r / (sqrt(2) sigma)) / r
print '(A)', '   i       r     phi       exact     rel. error'
b = 1 + (NB - 1) + NB * (NB/2 - 1) + NB**2 * (NB/2 - 1) ! the block of x >= 0.75 holding cell (j, k) = (16, 16)
do ig=(NB - 1)*NC + 1, NB*NC + NG, 3
   i = ig - (NB - 1) * NC
   x = ([ig, 16, 16] - 0.5_R_P) * H
   r = norm2(x - C)
   exact = -mass * erf(r / (sqrt(2._R_P) * SIGMA)) / r
   print '(I4,F8.4,2F10.4,ES11.1)', ig, r, phi(i,NC,NC,b), exact, abs(phi(i,NC,NC,b) / exact - 1._R_P)
enddo
