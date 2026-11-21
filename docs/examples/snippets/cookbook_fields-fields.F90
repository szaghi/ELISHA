! your density, ghosts and all: rho(1-NG:NC+NG, 1-NG:NC+NG, 1-NG:NC+NG, blocks); a mean of 1 under a wave
allocate(rho(1-NG:NC+NG,1-NG:NC+NG,1-NG:NC+NG,mesh%nb), phi(1-NG:NC+NG,1-NG:NC+NG,1-NG:NC+NG,mesh%nb))
rho = huge(1._R_P)                                         ! the ghosts: the solver never reads them
do b=1, mesh%nb ; do k=1, NC ; do j=1, NC ; do i=1, NC
   x = (mesh%coords(1:3,b) * NC + [i, j, k] - 0.5_R_P) * H
   rho(i,j,k,b) = 1._R_P + 0.5_R_P * sin(2 * PI * x(1)) * cos(2 * PI * x(2))
enddo ; enddo ; enddo ; enddo
