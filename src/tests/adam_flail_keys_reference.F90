!< Test oracle: ADAM's cell keys and hash table, verbatim from `adam_poisson_mg_object` (ADAM 669d0bdd).
module adam_flail_keys_reference
!< Test oracle: ADAM's cell keys and hash table, verbatim from `adam_poisson_mg_object` (ADAM 669d0bdd).
!<
!< The type, the constant and the procedures are ADAM's, line for line (private there, public here). Not part of the
!< library: never used by `src/lib`.
use :: penf, only : I4P, I8P
implicit none
private
public :: KEY_LEVEL_OFFSET, hash_table, bkey, ckey, kdecode, hash_init, hash_free, hash_put, hash_get

integer(I4P), parameter :: KEY_LEVEL_OFFSET=16_I4P !< Level offset in the cell keys (the levels below lmin, E5c).

type :: hash_table
   !< Open-addressing hash, int64 key (>= 0) -> int32 value (0: absent).
   integer(I8P), allocatable :: keys(:) !< Keys (-1: empty).
   integer(I4P), allocatable :: vals(:) !< Values.
   integer(I4P)              :: cap=0   !< Capacity (a prime).
endtype hash_table

contains
   pure function bkey(c4) result(key)
   !< Key of a block or cell: (i, j, k, level) packed in 19 + 19 + 19 + 7 bits.
   integer(I4P), intent(in) :: c4(4) !< Coordinates (i, j, k, level).
   integer(I8P)             :: key   !< Key.

   key = int(c4(4), I8P) * 2_I8P**57 + int(c4(1), I8P) * 2_I8P**38 + int(c4(2), I8P) * 2_I8P**19 + int(c4(3), I8P)
   endfunction bkey

   pure function ckey(lv, cc) result(key)
   !< Key of a cell at level lv.
   integer(I4P), intent(in) :: lv    !< Level.
   integer(I4P), intent(in) :: cc(3) !< Cell coordinates.
   integer(I8P)             :: key   !< Key.

   key = bkey([cc, lv + KEY_LEVEL_OFFSET]) ! the levels inside the blocks may be negative (#84 E5c)
   endfunction ckey

   pure subroutine kdecode(key, lv, cc)
   !< Decode a cell key.
   integer(I8P), intent(in)  :: key   !< Key.
   integer(I4P), intent(out) :: lv    !< Level.
   integer(I4P), intent(out) :: cc(3) !< Cell coordinates.

   lv = int(key / 2_I8P**57, I4P) - KEY_LEVEL_OFFSET
   cc(1) = int(mod(key / 2_I8P**38, 2_I8P**19), I4P)
   cc(2) = int(mod(key / 2_I8P**19, 2_I8P**19), I4P)
   cc(3) = int(mod(key, 2_I8P**19), I4P)
   endsubroutine kdecode

   subroutine hash_init(h, n)
   !< A table for about n keys (capacity: the first prime above 2 n).
   type(hash_table), intent(inout) :: h        !< Table.
   integer(I4P),     intent(in)    :: n        !< Expected keys.
   integer(I4P)                    :: c, f     !< Capacity candidate, factor.
   logical                         :: is_prime !< Primality.

   c = max(2 * n + 1, 7)
   do
      is_prime = .true.
      do f=2, int(sqrt(real(c))) + 1
         if (f < c .and. mod(c, f) == 0) then
            is_prime = .false.
            exit
         endif
      enddo
      if (is_prime) exit
      c = c + 1
   enddo
   call hash_free(h)
   h%cap = c
   allocate(h%keys(0:c-1), h%vals(0:c-1))
   h%keys = -1_I8P
   h%vals = 0
   endsubroutine hash_init

   subroutine hash_free(h)
   !< Free a table.
   type(hash_table), intent(inout) :: h !< Table.

   if (allocated(h%keys)) deallocate(h%keys, h%vals)
   h%cap = 0
   endsubroutine hash_free

   subroutine hash_put(h, key, val)
   !< Insert or overwrite (linear probing).
   type(hash_table), intent(inout) :: h   !< Table.
   integer(I8P),     intent(in)    :: key !< Key (>= 0).
   integer(I4P),     intent(in)    :: val !< Value (> 0).
   integer(I4P)                    :: s   !< Slot.

   s = int(modulo(key, int(h%cap, I8P)), I4P)
   do
      if (h%keys(s) == -1_I8P .or. h%keys(s) == key) exit
      s = mod(s + 1, h%cap)
   enddo
   h%keys(s) = key
   h%vals(s) = val
   endsubroutine hash_put

   pure function hash_get(h, key) result(val)
   !< The value of a key, 0 when absent.
   type(hash_table), intent(in) :: h    !< Table.
   integer(I8P),     intent(in) :: key  !< Key.
   integer(I4P)                 :: val  !< Value.
   integer(I4P)                 :: s, n !< Slot, probes.

   val = 0
   if (h%cap == 0) return
   s = int(modulo(key, int(h%cap, I8P)), I4P)
   do n=1, h%cap
      if (h%keys(s) == -1_I8P) return
      if (h%keys(s) == key) then
         val = h%vals(s)
         return
      endif
      s = mod(s + 1, h%cap)
   enddo
   endfunction hash_get

endmodule adam_flail_keys_reference
