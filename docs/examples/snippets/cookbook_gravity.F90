!< Cookbook: self-gravity. Poisson's equation for the potential of a density, lap Phi = 4 pi G rho, with phi -> 0 at
!< infinity, and the gravitational binding energy from the solver's energy: W = 1/2 sum rho Phi dV.
program cookbook_gravity
use elisha
implicit none
real(R_P), parameter :: PI=acos(-1._R_P), G=6.674e-8_R_P   ! cgs, for the factor; the check is in units of G
real(R_P), parameter :: SIGMA=0.08_R_P, C=0.5_R_P          ! a Gaussian cloud at the centre of [0, 1]^3
type(elisha_comm_object)  :: comm
integer(I4P)              :: stat, nc
real(R_P)                 :: w, mass
character(:), allocatable :: errmsg

call comm%init(0, stat, errmsg) ; call check
print '(A)', '   cells    W / (G M^2 / sigma)   exact -1/(2 sqrt(pi))   rel. error'
do nc=4, 8, 4
   call binding_energy(nc, w, mass)
   print '(I5,A,F18.6,F20.6,ES17.2)', 4*nc, '^3', w / (G * mass**2 / SIGMA), -0.5_R_P / sqrt(PI), &
                                       abs(w / (-G * mass**2 / (2 * sqrt(PI) * SIGMA)) - 1._R_P)
enddo
call comm%destroy(stat)
contains
   subroutine binding_energy(nc, w, mass)
   !< The binding energy and the mass of the cloud on 4^3 blocks of nc^3 cells.
   integer(I4P), intent(in)  :: nc   !< Cells per block per axis.
   real(R_P),    intent(out) :: w    !< Binding energy.
   real(R_P),    intent(out) :: mass !< Mass (the discrete sum).
   integer(I4P), parameter    :: NB=4, NG=1
   type(elisha_mesh_object)   :: mesh
   type(elisha_solver_object) :: solver
   integer(I4P)               :: coords(4,NB**3), b, i, j, k, n
   real(R_P), allocatable     :: rho(:), phi(:,:,:,:)
   real(R_P)                  :: h, x(3), energy
   h = 1._R_P / (NB * nc)
   b = 0
   do k=0, NB-1 ; do j=0, NB-1 ; do i=0, NB-1
      b = b + 1 ; coords(:,b) = [i, j, k, 0]
   enddo ; enddo ; enddo
   call mesh%init(nijk=[nc, nc, nc], ngc=NG, nblk0=[NB, NB, NB], dx0=[h, h, h], emin=[0._R_P, 0._R_P, 0._R_P], &
                  periodic=[.false., .false., .false.], null=[.false., .false., .false.], coords=coords, stat=stat, &
                  errmsg=errmsg)
   call check
   allocate(rho(mesh%nb * nc**3), phi(1-NG:nc+NG,1-NG:nc+NG,1-NG:nc+NG,mesh%nb))
   n = 0
   do b=1, mesh%nb ; do k=1, nc ; do j=1, nc ; do i=1, nc
      n = n + 1 ; x = (mesh%coords(1:3,b) * nc + [i, j, k] - 0.5_R_P) * h
      rho(n) = exp(-sum((x - C)**2) / (2 * SIGMA**2))
   enddo ; enddo ; enddo ; enddo
   mass = sum(rho) * h**3
   ! factor = 4 pi G: phi is the gravitational potential; energy = sum_cells rho phi dV, twice the binding energy
   call solver%setup(mesh, comm, stat, errmsg, isolated=.true.)
   call check
   call solver%solve_packed(mesh, rho, 4._R_P * PI * G, phi, stat, errmsg, energy=energy)
   call check
   w = 0.5_R_P * energy
   call solver%destroy
   call mesh%destroy
   endsubroutine binding_energy

   subroutine check
   !< Stop on a failed call, with the message.
   if (stat /= ELISHA_OK) then
      print '(A)', errmsg
      error stop 1
   endif
   endsubroutine check
endprogram cookbook_gravity
