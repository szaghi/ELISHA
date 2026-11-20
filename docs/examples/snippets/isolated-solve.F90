! L_h phi = 4 pi f with phi -> 0 at infinity: Hockney and Eastwood on the spectral path (a uniform mesh)
call solver%setup(mesh, comm, stat, errmsg, isolated=.true.)
call check
call solver%solve_packed(mesh, f, FACTOR, phi, stat, errmsg)
call check
