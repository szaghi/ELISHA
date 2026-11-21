! y and z are null axes (one cell, one block): their faces are periodic, so the problem is the line along x
bc(:,2:3) = ELISHA_BC_PERIODIC
select case(pair)
case(1) ; bc(:,1) = [ELISHA_BC_PERIODIC,  ELISHA_BC_PERIODIC]  ; k = 2._R_P * PI ! u = sin(2 pi x + 1)
case(2) ; bc(:,1) = [ELISHA_BC_DIRICHLET, ELISHA_BC_DIRICHLET] ; k = PI          ! u = sin(pi x)
case(3) ; bc(:,1) = [ELISHA_BC_NEUMANN,   ELISHA_BC_NEUMANN]   ; k = PI          ! u = cos(pi x)
case(4) ; bc(:,1) = [ELISHA_BC_DIRICHLET, ELISHA_BC_NEUMANN]   ; k = PI / 2      ! u = sin(pi x / 2)
endselect
call mesh%init(nijk=[nc, 1, 1], ngc=NG, nblk0=[NB, 1, 1], dx0=[h, 1._R_P, 1._R_P], emin=[0._R_P, 0._R_P, 0._R_P], &
               periodic=[pair == 1, .true., .true.], null=[.false., .true., .true.], coords=coords, stat=stat, &
               errmsg=errmsg)
call check
call solver%setup(mesh, comm, stat, errmsg, bc=bc) ! one level: the spectral path, an odd/even extension per kind
call check
