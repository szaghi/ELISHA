! collective: every rank calls setup and solve_packed with its own blocks; the energy comes back on every rank
call solver%setup(mesh, comm, stat, errmsg, tolerance=1.e-10_R_P)
call check
call solver%solve_packed(mesh, f, 1._R_P, phi, stat, errmsg, energy=energy)
call check
diag = solver%diagnostics()
