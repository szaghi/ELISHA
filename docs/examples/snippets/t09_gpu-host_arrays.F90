! host arrays: the backend copies the source in and the potential out
call solver%solve_packed(mesh, f, 1._R_P, phi_host, stat, errmsg)
