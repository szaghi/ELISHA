! and the periodic solve of the same star: the potential of a lattice of stars, mean density removed
call solver%destroy
call mesh%destroy
call mesh%init(nijk=[NC, NC, NC], ngc=NG, nblk0=[NB, NB, NB], dx0=[H, H, H], emin=[0._R_P, 0._R_P, 0._R_P], &
               periodic=[.true., .true., .true.], null=[.false., .false., .false.], coords=coords, stat=stat, &
               errmsg=errmsg)
call check
call solver%setup(mesh, comm, stat, errmsg)
call check
call solver%solve_packed(mesh, f, FOUR_PI_G, phi(:,:,:,:,3), stat, errmsg)
call check
