! z is a null axis: one cell, one block; the octree is a quadtree
call mesh%init(nijk=[NC, NC, 1], ngc=NG, nblk0=[NB, NB, 1], dx0=[H, H, 1._R_P], emin=[0._R_P, 0._R_P, 0._R_P], &
               periodic=[.true., .true., .true.], null=[.false., .false., .true.], coords=coords(:,1:nleaf), &
               stat=stat, errmsg=errmsg)
call check
