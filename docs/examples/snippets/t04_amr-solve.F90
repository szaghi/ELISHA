! auto: the spectral path on a uniform mesh, the hierarchical one (multigrid FCG, the James step) on the AMR mesh
call solver%setup(mesh, comm, stat, errmsg, isolated=.true., tolerance=1.e-10_R_P)
call check
call solver%solve_packed(mesh, f, FOUR_PI_G, phi, stat, errmsg)
call check
diag = solver%diagnostics()
