! L_h phi = 4 pi G rho with phi -> 0 at infinity (G = 1): on a uniform mesh, auto chooses the spectral path
call solver%setup(mesh, comm, stat, errmsg, isolated=.true.)
call check
call solver%solve_packed(mesh, f, FOUR_PI_G, phi(:,:,:,:,1), stat, errmsg)
call check
! the same with the hierarchical path forced: the James step, two multigrid solves and a boundary convolution
call solver%setup(mesh, comm, stat, errmsg, isolated=.true., path=ELISHA_PATH_HIERARCHICAL, tolerance=1.e-12_R_P)
call check
call solver%solve_packed(mesh, f, FOUR_PI_G, phi(:,:,:,:,2), stat, errmsg)
call check
