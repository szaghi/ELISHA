allocate(f(mesh%nb * nc**3), phi(1-NG:nc+NG,1-NG:nc+NG,1-NG:nc+NG,mesh%nb))
f = 0._R_P                                                    ! Laplace: no source
call solver%solve_packed(mesh, f, 1._R_P, phi, stat, errmsg, ghost=g)
call check
diag = solver%diagnostics() ; iters = diag%iterations
