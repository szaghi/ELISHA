! device-resident arrays: f and phi are present on the device, and the solve stages nothing (copy, not
! copyout: the ghosts the solve does not write keep their values)
!$acc data copyin(f) copy(phi_dev)
call solver%solve_packed(mesh, f, 1._R_P, phi_dev, stat, errmsg, on_device=ON_DEVICE)
!$acc end data
