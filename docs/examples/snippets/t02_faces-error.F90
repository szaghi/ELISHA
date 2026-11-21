! where no Dirichlet face fixes the constant, phi is defined up to one: compare with the means matched
c = merge(0._R_P, (sum(phi(1:nc,1,1,:)) - sum(u)) / (NB * nc), pair == 2 .or. pair == 4)
err = maxval(abs(pack(phi(1:nc,1,1,:), .true.) - c - u)) / maxval(abs(u))
