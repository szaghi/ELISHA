! isolated boundaries need every axis non-periodic and cubic cells
call mesh%init(nijk=[NC, NC, NC], ngc=NG, nblk0=[NB, NB, NB], dx0=[H, H, H], emin=[0._R_P, 0._R_P, 0._R_P], &
               periodic=[.false., .false., .false.], null=[.false., .false., .false.], coords=coords, stat=stat, &
               errmsg=errmsg)
call check
