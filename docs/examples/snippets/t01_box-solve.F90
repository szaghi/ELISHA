call solver%setup(mesh, comm, stat, errmsg)              ! path auto: one level, so the spectral solver
call check
call solver%solve_packed(mesh, f, 1._R_P, phi, stat, errmsg) ! L_h phi = f - mean f, every ghost layer filled
call check
