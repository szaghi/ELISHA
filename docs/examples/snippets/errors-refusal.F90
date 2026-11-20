! a solve before any setup
call solver%solve_packed(mesh, f, 1._R_P, phi, stat, errmsg)
call report
! the spectral path asked on a mesh at two levels: refused, and the solver is left not set up
call solver%setup(mesh, comm, stat, errmsg, path=ELISHA_PATH_SPECTRAL)
call report
print '(A,I0)', 'path after the refusal: ', solver%path()
! the caller's choice: fall back to the automatic path
call solver%setup(mesh, comm, stat, errmsg, path=ELISHA_PATH_AUTO)
call report
