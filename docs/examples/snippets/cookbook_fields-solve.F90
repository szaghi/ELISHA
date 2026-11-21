call solver%setup(mesh, comm, stat, errmsg)
call check
call solver%solve_packed(mesh, buf, 1._R_P, phi, stat, errmsg) ! periodic: L_h phi = rho - mean rho
call check
