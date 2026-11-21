! Dirichlet on every face: boundary values need the hierarchical path (here chosen by the two levels anyway)
bc = ELISHA_BC_DIRICHLET
call solver%setup(mesh, comm, stat, errmsg, bc=bc, tolerance=1.e-12_R_P)
call check
