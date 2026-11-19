!< Test shim: stands in for ADAM's `adam_field_object`, just enough for the verbatim `adam_poisson_fft_object`.
module adam_field_object
!< Test shim: stands in for ADAM's `adam_field_object`, just enough for the verbatim `adam_poisson_fft_object`.
!<
!< Not ADAM's object: a plain container with the members `adam_poisson_fft_object` (ADAM 669d0bdd) reads, and nothing
!< else. The solver reads, through `field%`:
!< - `blocks_number`: this rank's blocks;
!< - `coordinates(4,b)`: block b's (i, j, k), 0-based at the block's level, then the level (ADAM's convention, the
!<   one `elisha_mesh_object%coords` keeps).
!< It reads no `emin`/`emax` of the field. Not part of the library: never used by `src/lib`.
use :: penf, only : I4P
implicit none
private
public :: field_object

type :: field_object
   !< ADAM's field, the members the FFT solver reads.
   integer(I4P)              :: blocks_number=0_I4P !< This rank's blocks.
   integer(I4P), allocatable :: coordinates(:,:)    !< This rank's blocks (i, j, k, level) [4,blocks_number].
   contains
      procedure, pass(self) :: set !< Fill the field from this rank's block coordinates.
endtype field_object

contains
   subroutine set(self, coords)
   !< Fill the field from this rank's block coordinates (i, j, k, level) [4,nb].
   class(field_object), intent(inout) :: self        !< The field.
   integer(I4P),        intent(in)    :: coords(:,:) !< Block coordinates [4,nb].

   if (allocated(self%coordinates)) deallocate(self%coordinates)
   self%blocks_number = size(coords, dim=2, kind=I4P)
   allocate(self%coordinates(4,max(self%blocks_number,1_I4P)))
   self%coordinates = 0_I4P
   if (self%blocks_number > 0) self%coordinates(:,1:self%blocks_number) = coords
   endsubroutine set
endmodule adam_field_object
