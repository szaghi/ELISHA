! a fresh setup per k (a solve warm-starts from the previous one of its setup): each solve starts from phi = 0
history = 0._R_P ; iters = 0
do cycle_kind=1, 2
   do k=1, KMAX
      call solver%setup(mesh, comm, stat, errmsg, max_iterations=k, tolerance=1.e-10_R_P, &
                        vcycle=merge(ELISHA_VCYCLE_SYMMETRIC, ELISHA_VCYCLE_ADAM, cycle_kind == 1))
      call check
      call solver%solve_packed(mesh, f, 1._R_P, phi, stat, errmsg)
      diag = solver%diagnostics()
      history(k,cycle_kind) = diag%residual
      if (dump) frames(:,:,:,k,cycle_kind) = phi(1:NC,1:NC,1,:)
      if (stat == ELISHA_OK) then                  ! converged within k iterations
         iters(cycle_kind) = diag%iterations
         exit
      elseif (stat /= ELISHA_ERR_NOT_CONVERGED) then ! a real failure
         call check
      endif                                        ! not converged: phi is the k-th iterate, its ghosts filled
   enddo
enddo
