! a Gaussian star at the centre, 7 sigma inside every face; its mass is the discrete sum
allocate(f(mesh%nb * NC**3), phi(1-NG:NC+NG,1-NG:NC+NG,1-NG:NC+NG,mesh%nb,3))
n = 0
do b=1, mesh%nb ; do k=1, NC ; do j=1, NC ; do i=1, NC
   n = n + 1 ; x = (mesh%coords(1:3,b) * NC + [i, j, k] - 0.5_R_P) * H
   f(n) = exp(-sum((x - C)**2) / (2._R_P * SIGMA**2))
enddo ; enddo ; enddo ; enddo
mass = sum(f) * H**3
