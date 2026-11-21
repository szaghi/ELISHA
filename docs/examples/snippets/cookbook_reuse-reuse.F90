call warm%setup(mesh, comm, stat, errmsg, tolerance=1.e-10_R_P) ! once: the mesh does not change
call check
print '(A)', '  step   FCG, one setup   FCG, a setup per step'
do step=1, NSTEP
   call source(step)
   call warm%solve_packed(mesh, f, 1._R_P, phi_warm, stat, errmsg) ! starts from the previous step's potential
   call check
   call cold%setup(mesh, comm, stat, errmsg, tolerance=1.e-10_R_P) ! starts from zero
   call check
   call cold%solve_packed(mesh, f, 1._R_P, phi_cold, stat, errmsg)
   call check
   dw = warm%diagnostics() ; dc = cold%diagnostics()
   print '(I6,I14,I21)', step, dw%iterations, dc%iterations
enddo
