! the leaves of the octree: a block near the star is replaced by its 8 children, down to level lmax
allocate(coords(4,NB**3 * 8**lmax)) ; nleaf = 0
do k=0, NB-1 ; do j=0, NB-1 ; do i=0, NB-1
   call refine([i, j, k, 0])
enddo ; enddo ; enddo
call mesh%init(nijk=[NC, NC, NC], ngc=NG, nblk0=[NB, NB, NB], dx0=[H, H, H], emin=[0._R_P, 0._R_P, 0._R_P], &
               periodic=[.false., .false., .false.], null=[.false., .false., .false.], coords=coords(:,1:nleaf), &
               stat=stat, errmsg=errmsg)
call check
