! a regrid to a 2:1 mesh, Dirichlet everywhere, no source and the values of u = 1 + x + 2 y - z on the faces: the
! compact operator is exact for linear u, across the 2:1 seam too, so phi = u at every cell centre
call make_mesh(refined=.true.)
bc = ELISHA_BC_DIRICHLET
call solver%setup(mesh, comm, stat, errmsg, bc=bc, tolerance=1.e-12_R_P) ! two levels: the hierarchical path
call check
call solver%boundary_faces(kinds, dir, xf, stat, errmsg) ! this rank's faces: kind, outward axis, where the value is
call check
g = 1._R_P + xf(1,:) + 2._R_P * xf(2,:) - xf(3,:)
f = 0._R_P
call solver%solve_packed(mesh, f, 1._R_P, phi, stat, errmsg, ghost=g)
call check
err = 0._R_P
do b=1, mesh%nb ; do k=1, NC ; do j=1, NC ; do i=1, NC
   x = (mesh%coords(1:3,b) * NC + [i, j, k] - 0.5_R_P) * H / 2**mesh%coords(4,b)
   err = max(err, abs(phi(i,j,k,b) - (1._R_P + x(1) + 2._R_P * x(2) - x(3))))
enddo ; enddo ; enddo ; enddo
print '(A,L1,A,I0,A,L1)', 'hierarchical ', solver%is_hierarchical(), ', boundary faces ', size(g), &
                          ', phi = u to 1e-9 ', err <= 1.e-9_R_P
