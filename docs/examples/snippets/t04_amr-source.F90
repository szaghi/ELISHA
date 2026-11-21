! the star at the cell centres of each block's own level: a block at level l has spacing H / 2^l
allocate(f(mesh%nb * NC**3), phi(1-NG:NC+NG,1-NG:NC+NG,1-NG:NC+NG,mesh%nb))
n = 0 ; mass = 0._R_P
do b=1, mesh%nb
   hb = H / 2**mesh%coords(4,b)
   do k=1, NC ; do j=1, NC ; do i=1, NC
      n = n + 1 ; x = (mesh%coords(1:3,b) * NC + [i, j, k] - 0.5_R_P) * hb
      f(n) = exp(-sum((x - C)**2) / (2._R_P * SIGMA**2)) ; mass = mass + f(n) * hb**3
   enddo ; enddo ; enddo
enddo
