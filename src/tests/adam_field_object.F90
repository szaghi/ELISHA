!< Test shim: stands in for ADAM's `adam_field_object`, just enough for the verbatim ADAM Poisson solvers.
module adam_field_object
!< Test shim: stands in for ADAM's `adam_field_object`, just enough for the verbatim ADAM Poisson solvers.
!<
!< Not ADAM's object: a plain container with the members the verbatim solvers of ADAM 669d0bdd read, and nothing else.
!< Through `field%`:
!< - `blocks_number`: this rank's blocks (`adam_poisson_fft_object`, `adam_poisson_mg_object`);
!< - `coordinates(4,b)`: block b's (i, j, k), 0-based at the block's level, then the level (ADAM's convention, the
!<   one `elisha_mesh_object%coords` keeps) (`adam_poisson_fft_object`, `adam_poisson_mg_object`);
!< - `emin(3,b)`, `emax(3,b)`: block b's corners (`adam_poisson_mg_object` only: the centre of a null axis, for the
!<   position of a domain-face value; zero unless `set` is given them).
!< Not part of the library: never used by `src/lib`.
use :: penf, only : I4P, R8P
implicit none
private
public :: field_object

type :: field_object
   !< ADAM's field, the members the Poisson solvers read.
   integer(I4P)              :: blocks_number=0_I4P !< This rank's blocks.
   integer(I4P), allocatable :: coordinates(:,:)    !< This rank's blocks (i, j, k, level) [4,blocks_number].
   real(R8P),    allocatable :: emin(:,:)           !< This rank's block lower corners [3,blocks_number].
   real(R8P),    allocatable :: emax(:,:)           !< This rank's block upper corners [3,blocks_number].
   contains
      procedure, pass(self) :: set !< Fill the field from this rank's block coordinates (and corners).
endtype field_object

contains
   subroutine set(self, coords, emin, emax)
   !< Fill the field from this rank's block coordinates (i, j, k, level) [4,nb], and the block corners [3,nb] when
   !< given (zero otherwise).
   class(field_object), intent(inout)        :: self        !< The field.
   integer(I4P),        intent(in)           :: coords(:,:) !< Block coordinates [4,nb].
   real(R8P),           intent(in), optional :: emin(:,:)   !< Block lower corners [3,nb].
   real(R8P),           intent(in), optional :: emax(:,:)   !< Block upper corners [3,nb].

   if (allocated(self%coordinates)) deallocate(self%coordinates)
   if (allocated(self%emin)) deallocate(self%emin)
   if (allocated(self%emax)) deallocate(self%emax)
   self%blocks_number = size(coords, dim=2, kind=I4P)
   allocate(self%coordinates(4,max(self%blocks_number,1_I4P)))
   allocate(self%emin(3,max(self%blocks_number,1_I4P)), self%emax(3,max(self%blocks_number,1_I4P)))
   self%coordinates = 0_I4P
   self%emin = 0._R8P
   self%emax = 0._R8P
   if (self%blocks_number > 0) then
      self%coordinates(:,1:self%blocks_number) = coords
      if (present(emin)) self%emin(:,1:self%blocks_number) = emin
      if (present(emax)) self%emax(:,1:self%blocks_number) = emax
   endif
   endsubroutine set
endmodule adam_field_object
