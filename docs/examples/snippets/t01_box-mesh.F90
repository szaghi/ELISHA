h = 1._R_P / (NB * nc)                    ! cell spacing
b = 0
do k=0, NB-1 ; do j=0, NB-1 ; do i=0, NB-1 ! this rank's blocks: (i, j, k) at level 0, then the level
   b = b + 1 ; coords(:,b) = [i, j, k, 0]
enddo ; enddo ; enddo
call mesh%init(nijk=[nc, nc, nc], ngc=NG, nblk0=[NB, NB, NB], dx0=[h, h, h], emin=[0._R_P, 0._R_P, 0._R_P], &
               periodic=[.true., .true., .true.], null=[.false., .false., .false.], coords=coords, stat=stat, &
               errmsg=errmsg)
call check
