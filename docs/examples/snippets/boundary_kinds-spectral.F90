! x: Dirichlet at the low face, Neumann at the high one; y: the reverse; z: Dirichlet at both (homogeneous values)
bc(:,1) = [ELISHA_BC_DIRICHLET, ELISHA_BC_NEUMANN]
bc(:,2) = [ELISHA_BC_NEUMANN, ELISHA_BC_DIRICHLET]
bc(:,3) = [ELISHA_BC_DIRICHLET, ELISHA_BC_DIRICHLET]
call make_mesh(refined=.false.)
call solver%setup(mesh, comm, stat, errmsg, bc=bc) ! one level: the spectral path (odd/even extensions)
call check
call solver%solve_packed(mesh, f, 1._R_P, phi, stat, errmsg)
call check
! the ghost layer carries the kinds: phi_ghost = -phi_boundary at a Dirichlet face, = phi_boundary at a Neumann one
d_lo = 0._R_P ; n_hi = 0._R_P
do b=1, mesh%nb
   if (mesh%coords(1,b) == 0)      d_lo = max(d_lo, maxval(abs(phi(0,1:NC,1:NC,b) + phi(1,1:NC,1:NC,b))))
   if (mesh%coords(1,b) == NB - 1) n_hi = max(n_hi, maxval(abs(phi(NC+1,1:NC,1:NC,b) - phi(NC,1:NC,1:NC,b))))
enddo
print '(A,L1,A,L1)', 'spectral ', solver%path() == ELISHA_PATH_SPECTRAL, ', x low Dirichlet ', &
                     d_lo <= 1.e-12_R_P * maxval(abs(phi))
print '(A,L1)', 'x high Neumann ', n_hi <= 1.e-12_R_P * maxval(abs(phi))
