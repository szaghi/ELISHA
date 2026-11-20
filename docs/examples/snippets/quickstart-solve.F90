call solver%setup(mesh, comm, stat, errmsg)                  ! path auto: one level, so the spectral solver
call check
call solver%solve_packed(mesh, f, FACTOR, phi, stat, errmsg) ! L_h phi = factor (f - mean f), ghosts filled
call check
diag = solver%diagnostics()
print '(A,L1,A,I0)', 'spectral path ', solver%path() == ELISHA_PATH_SPECTRAL, ', iterations ', diag%iterations
