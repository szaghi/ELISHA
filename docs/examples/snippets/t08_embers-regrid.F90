! a new mesh around the embers' new places, and a new setup on it (setup destroys the previous solver)
nleaf = 0
do k=0, NB-1 ; do j=0, NB-1 ; do i=0, NB-1
   call refine([i, j, k, 0])
enddo ; enddo ; enddo
call mesh%destroy
call mesh%init(nijk=[NC, NC, NC], ngc=NG, nblk0=[NB, NB, NB], dx0=[H, H, H], emin=[0._R_P, 0._R_P, 0._R_P], &
               periodic=[.false., .false., .false.], null=[.false., .false., .false.], coords=coords(:,1:nleaf), &
               stat=stat, errmsg=errmsg)
call check
call solver%setup(mesh, comm, stat, errmsg, isolated=.true., tolerance=1.e-10_R_P)
call check
