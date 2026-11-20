call solver%setup(mesh, comm, stat, errmsg, tolerance=1.e-10_R_P) ! path auto: two levels, so hierarchical
call check
call solver%solve_packed(mesh, f, FACTOR, phi, stat, errmsg, energy=energy) ! energy: sum_cells f phi dV
call check
diag = solver%diagnostics()
print '(A,L1,A,I0,A,L1)', 'hierarchical ', solver%is_hierarchical(), ', FCG iterations ', diag%iterations, &
                          ', converged ', diag%residual <= diag%tolerance_used
print '(A,ES11.4)', 'energy ', energy
