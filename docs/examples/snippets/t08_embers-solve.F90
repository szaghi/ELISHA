call solver%solve_packed(mesh, f, FOUR_PI_G, phi, stat, errmsg, energy=energy)
call check
diag = solver%diagnostics()
