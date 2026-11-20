! block (0,0,0) is replaced by its 8 children at level 1, coordinates (2i+o_x, 2j+o_y, 2k+o_z) at that level
b = 0
do k=0, NB-1 ; do j=0, NB-1 ; do i=0, NB-1
   if (all([i, j, k] == 0)) then
      do o=0, 7
         b = b + 1 ; coords(:,b) = [2*i + ibits(o, 0, 1), 2*j + ibits(o, 1, 1), 2*k + ibits(o, 2, 1), 1]
      enddo
   else
      b = b + 1 ; coords(:,b) = [i, j, k, 0]
   endif
enddo ; enddo ; enddo
call mesh%init(nijk=[NC, NC, NC], ngc=NG, nblk0=[NB, NB, NB], dx0=[H, H, H], emin=[0._R_P, 0._R_P, 0._R_P], &
               periodic=[.true., .true., .true.], null=[.false., .false., .false.], coords=coords, stat=stat, &
               errmsg=errmsg)
call check
