! factor = 4 pi G: phi is the gravitational potential; energy = sum_cells rho phi dV, twice the binding energy
call solver%setup(mesh, comm, stat, errmsg, isolated=.true.)
call check
call solver%solve_packed(mesh, rho, 4._R_P * PI * G, phi, stat, errmsg, energy=energy)
call check
w = 0.5_R_P * energy
