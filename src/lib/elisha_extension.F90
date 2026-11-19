!< ELISHA, L0 extension maps: the source cell and sign of every cell of an axis's transform array.
module elisha_extension
!< ELISHA, L0 extension maps: the source cell and sign of every cell of an axis's transform array.
!<
!< The spectral solver turns face kinds into a periodic problem by extending the source over the transform cells of
!< each axis (`elisha_axis_transform_size`): odd about a Dirichlet face, even about a Neumann one, odd about the ghost
!< centres for ghost-Dirichlet, zero padding when isolated. `elisha_axis_map` gives, for a target cell t (0-based) of
!< the transform array, the cell of the domain it copies (0-based, -1 for a zero) and the sign; `elisha_set_plane`
!< applies one such map to a plane of a 3-D array. Ported from ADAM's `adam_poisson_fft_object` (ADAM 669d0bdd):
!< `axis_map` was an internal procedure reading the solver's state (`self%n`, `self%m`, `self%bc`, `self%isolated`),
!< here those are arguments; the arithmetic is unchanged.
use penf, only : I4P
use elisha_kind_parameters, only : R_P
use elisha_parameters, only : ELISHA_BC_DIRICHLET, ELISHA_BC_DIRICHLET_GHOST
implicit none
private
public :: elisha_axis_map
public :: elisha_set_plane

contains
   pure recursive subroutine elisha_axis_map(n, m, bc, isolated, t, src, sg)
   !< Target cell t (0-based) of an axis of n cells extended to m: its source cell (0-based, -1 for zero) and sign.
   integer(I4P), intent(in)  :: n        !< Cells of the domain along the axis.
   integer(I4P), intent(in)  :: m        !< Transform cells along the axis.
   integer(I4P), intent(in)  :: bc(2)    !< Kinds of the low and high faces.
   logical,      intent(in)  :: isolated !< Isolated boundaries (zero padding).
   integer(I4P), intent(in)  :: t        !< Target cell.
   integer(I4P), intent(out) :: src      !< Source cell.
   real(R_P),    intent(out) :: sg       !< Sign.

   src = t ; sg = 1._R_P
   if (t < n) return ! the domain itself
   src = -1 ; sg = 0._R_P
   if (isolated) return ! the padding
   if (all(bc == ELISHA_BC_DIRICHLET_GHOST)) then ! odd about the ghost centres n and m - 1
      if (t == n .or. t == m - 1) return
      src = 2 * n - t ; sg = -1._R_P
   elseif (t < 2 * n) then ! the reflection about the high face
      src = 2 * n - 1 - t ; sg = merge(-1._R_P, 1._R_P, bc(2) == ELISHA_BC_DIRICHLET)
   else ! the reflection of [0, 2n) about the low face
      call elisha_axis_map(n, m, bc, isolated, 4 * n - 1 - t, src, sg)
      sg = sg * merge(-1._R_P, 1._R_P, bc(1) == ELISHA_BC_DIRICHLET)
   endif
   endsubroutine elisha_axis_map

   pure subroutine elisha_set_plane(f, d, t, src, sg)
   !< The plane t of axis d set to sg times the plane src (0-based cell indices; sg = 0 zeroes it).
   real(R_P),    intent(inout) :: f(1:,1:,1:) !< The array.
   integer(I4P), intent(in)    :: d, t, src   !< Axis, target and source cell indices.
   real(R_P),    intent(in)    :: sg          !< Sign (0: zero).

   select case(d)
   case(1)
      if (sg == 0._R_P) then
         f(t+1,:,:) = 0._R_P
      else
         f(t+1,:,:) = sg * f(src+1,:,:)
      endif
   case(2)
      if (sg == 0._R_P) then
         f(:,t+1,:) = 0._R_P
      else
         f(:,t+1,:) = sg * f(:,src+1,:)
      endif
   case(3)
      if (sg == 0._R_P) then
         f(:,:,t+1) = 0._R_P
      else
         f(:,:,t+1) = sg * f(:,:,src+1)
      endif
   endselect
   endsubroutine elisha_set_plane
endmodule elisha_extension
