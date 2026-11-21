! the source packed block-major, i fastest: f = sin(2 pi x) sin(2 pi y) sin(2 pi z) at the cell centres
allocate(f(mesh%nb * nc**3), phi(1-NG:nc+NG,1-NG:nc+NG,1-NG:nc+NG,mesh%nb))
n = 0
do b=1, mesh%nb ; do k=1, nc ; do j=1, nc ; do i=1, nc
   n = n + 1 ; x = (mesh%coords(1:3,b) * nc + [i, j, k] - 0.5_R_P) * h
   f(n) = product(sin(2._R_P * PI * x))
enddo ; enddo ; enddo ; enddo
