! the source packed block-major, i fastest: f = sin(2 pi x) sin(2 pi y) sin(2 pi z) at the cell centres
allocate(f(mesh%nb * NC**3), phi(1-NG:NC+NG,1-NG:NC+NG,1-NG:NC+NG,mesh%nb))
n = 0
do b=1, mesh%nb ; do k=1, NC ; do j=1, NC ; do i=1, NC
   n = n + 1 ; x = (mesh%coords(1:3,b) * NC + [i, j, k] - 0.5_R_P) * H
   f(n) = product(sin(2._R_P * PI * x))
enddo ; enddo ; enddo ; enddo
