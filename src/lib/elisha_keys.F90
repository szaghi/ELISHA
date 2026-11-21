!< ELISHA, L1 cell keys: a block or cell (i, j, k, level) packed in one int64, and an open-addressing table on them.
module elisha_keys
!< ELISHA, L1 cell keys: a block or cell (i, j, k, level) packed in one int64, and an open-addressing table on them.
!<
!< Ported from ADAM's `adam_poisson_mg_object` (ADAM 669d0bdd): the keys with the same arithmetic (`elisha_test_keys`
!< holds them to ADAM's bitwise), the table with ADAM's capacity, empty mark and probing but its own slot function
!< (below):
!< - `elisha_bkey([i, j, k, l])`: 19 + 19 + 19 bits for the coordinates (0 <= i, j, k < 2^19: `ELISHA_KEY_COORD_LIMIT`)
!<   and the level above them, `l * 2^57 + i * 2^38 + j * 2^19 + k`. ADAM's comment says 7 bits for the level, but the
!<   seventh is the sign: a level field of 64 or more overflows int64 (negative keys, and -1 is the table's empty
!<   mark), so the field is 0..63 and a cell level at most `ELISHA_KEY_LEVEL_MAX` = 63 - 16 = 47. Unreachable in
!<   practice: below 2^19 cells per axis the leaf level is at most 18 (the mesh's key-limit check);
!< - `elisha_ckey(lv, cc)`: a cell key, the level offset by `ELISHA_KEY_LEVEL_OFFSET` (16) since the multigrid's levels
!<   inside the blocks go below zero; `elisha_kdecode` inverts it;
!< - `elisha_hash_table`: int64 key (>= 0) -> int32 value (> 0; 0 means absent), linear probing, capacity the first
!<   prime above 2 n for n expected keys, -1 the empty mark (ADAM's). `put` needs a free slot: at most n distinct keys
!<   after `init(n)`.
!<
!< **The slot function is not ADAM's** (ELISHA M4). ADAM's first slot was `modulo(key, cap)`. A cell key is linear in
!< its coordinates, `l 2^57 + i 2^38 + j 2^19 + k`, so its slot is `(A + i (2^38 mod cap) + j (2^19 mod cap) + k) mod
!< cap`; when the capacity is a prime just above 2^19 (n of about 2^18 keys: 524 309 for n = 262 144), 2^19 mod cap is
!< -21, consecutive j land 21 slots apart and the runs of consecutive k merge into one cluster that linear probing walks
!< end to end. Measured: the 131 072 halo keys of the level-4 restriction of a 491 520-cell AMR mesh at np 2 (a 64 x 64
!< x 32 slab of cells) cost 50 967 probes per insert, 44.8 of the setup's 48.5 s (ADAM and ELISHA alike; 100-110 s at
!< np 4, 3 s at np 1, where that operator has no halo). The slot is now `elisha_hash_slot(key, cap)`: the key's 63 bits
!< mixed by a xorshift (`x ^= x << 13; x ^= x >> 7; x ^= x << 17`, each step kept to 63 bits, so every value is
!< non-negative and only `ieor`, `ishft` and `iand` touch it: no integer multiply, so no signed overflow on any
!< compiler), then `modulo(x, cap)`. The xorshift is a bijection of the 63-bit words, linear over GF(2) and not over the
!< integers, so it breaks the arithmetic lattice of the keys: the same slab costs 0.17 probes per insert, a full 64^3
!< box 0.51, random keys 0.51 (as a splitmix64 finaliser, measured alike, without its 64-bit multiplies).
!< No result depends on the slot positions: the table only maps a key to the value stored with it, `get` returns the
!< same value for every key whatever the slots, and the solvers number their entries (leaf, node and halo ids) in
!< first-seen order, not by slot. `elisha_test_keys` holds the key -> value map to ADAM's table on the same inserts and
!< guards the clustering; the hierarchical parity test holds every solver result bitwise to ADAM's (which keeps ADAM's
!< table).
use penf, only : I4P, I8P
implicit none
private
public :: ELISHA_KEY_COORD_LIMIT, ELISHA_KEY_LEVEL_OFFSET, ELISHA_KEY_LEVEL_MAX
public :: elisha_bkey, elisha_bkey4, elisha_ckey, elisha_kdecode, elisha_hash_slot
public :: elisha_hash_table

integer(I4P), parameter :: ELISHA_KEY_COORD_LIMIT=2_I4P**19 !< Coordinates are below it (19 bits).
integer(I4P), parameter :: ELISHA_KEY_LEVEL_OFFSET=16_I4P  !< Level offset in the cell keys (levels below lmin).
integer(I4P), parameter :: ELISHA_KEY_LEVEL_MAX=63_I4P - ELISHA_KEY_LEVEL_OFFSET !< Highest cell level (int64 sign).

type :: elisha_hash_table
   !< Open-addressing hash, int64 key (>= 0) -> int32 value (0: absent).
   integer(I8P), allocatable :: keys(:) !< Keys (-1: empty).
   integer(I4P), allocatable :: vals(:) !< Values.
   integer(I4P)              :: cap=0   !< Capacity (a prime).
   contains
      procedure, pass(h) :: init !< A table for about n keys.
      procedure, pass(h) :: free !< Free the table.
      procedure, pass(h) :: put  !< Insert or overwrite.
      procedure, pass(h) :: get  !< The value of a key, 0 when absent.
endtype elisha_hash_table

contains
   pure function elisha_bkey(c4) result(key)
   !< Key of a block or cell: (i, j, k, level) packed in 19 + 19 + 19 bits and the level above them (6 bits below the
   !< int64 sign: see the module; ADAM's comment says 7). The array form of `elisha_bkey4`.
   integer(I4P), intent(in) :: c4(4) !< Coordinates (i, j, k, level).
   integer(I8P)             :: key   !< Key.

   key = elisha_bkey4(c4(1), c4(2), c4(3), c4(4))
   endfunction elisha_bkey

   pure function elisha_bkey4(i, j, k, l) result(key)
   !< `elisha_bkey` on four scalars (ADAM's arithmetic): no array to build at the call, so no argument temporary
   !< (ifx `-check arg_temp_created` flags every array constructor passed as an argument, ELISHA notes decision 4).
   integer(I4P), intent(in) :: i, j, k !< Coordinates.
   integer(I4P), intent(in) :: l       !< Level.
   integer(I8P)             :: key     !< Key.

   key = int(l, I8P) * 2_I8P**57 + int(i, I8P) * 2_I8P**38 + int(j, I8P) * 2_I8P**19 + int(k, I8P)
   endfunction elisha_bkey4

   pure function elisha_ckey(lv, cc) result(key)
   !< Key of a cell at level lv.
   integer(I4P), intent(in) :: lv    !< Level.
   integer(I4P), intent(in) :: cc(3) !< Cell coordinates.
   integer(I8P)             :: key   !< Key.

   key = elisha_bkey4(cc(1), cc(2), cc(3), lv + ELISHA_KEY_LEVEL_OFFSET) ! levels inside the blocks may be negative (#84 E5c)
   endfunction elisha_ckey

   pure subroutine elisha_kdecode(key, lv, cc)
   !< Decode a cell key.
   integer(I8P), intent(in)  :: key   !< Key.
   integer(I4P), intent(out) :: lv    !< Level.
   integer(I4P), intent(out) :: cc(3) !< Cell coordinates.

   lv = int(key / 2_I8P**57, I4P) - ELISHA_KEY_LEVEL_OFFSET
   cc(1) = int(mod(key / 2_I8P**38, 2_I8P**19), I4P)
   cc(2) = int(mod(key / 2_I8P**19, 2_I8P**19), I4P)
   cc(3) = int(mod(key, 2_I8P**19), I4P)
   endsubroutine elisha_kdecode


   pure function elisha_hash_slot(key, cap) result(s)
   !< The first slot of a key in a table of `cap` slots: the key's 63 bits mixed by a xorshift, each step kept to 63
   !< bits (non-negative throughout: only bit operations, no overflow), then modulo the capacity (see the module).
   integer(I8P), intent(in) :: key !< Key (>= 0).
   integer(I4P), intent(in) :: cap !< Capacity (> 0).
   integer(I4P)             :: s   !< Slot, 0..cap-1.
   integer(I8P)             :: x   !< Mixed key.

   x = iand(key, huge(1_I8P))
   x = iand(ieor(x, ishft(x, 13)), huge(1_I8P))
   x = ieor(x, ishft(x, -7))
   x = iand(ieor(x, ishft(x, 17)), huge(1_I8P))
   s = int(modulo(x, int(cap, I8P)), I4P)
   endfunction elisha_hash_slot

   subroutine init(h, n)
   !< A table for about n keys (capacity: the first prime above 2 n).
   class(elisha_hash_table), intent(inout) :: h        !< Table.
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
   call h%free
   h%cap = c
   allocate(h%keys(0:c-1), h%vals(0:c-1))
   h%keys = -1_I8P
   h%vals = 0
   endsubroutine init

   subroutine free(h)
   !< Free a table.
   class(elisha_hash_table), intent(inout) :: h !< Table.

   if (allocated(h%keys)) deallocate(h%keys, h%vals)
   h%cap = 0
   endsubroutine free

   subroutine put(h, key, val)
   !< Insert or overwrite (linear probing).
   class(elisha_hash_table), intent(inout) :: h   !< Table.
   integer(I8P),     intent(in)    :: key !< Key (>= 0).
   integer(I4P),     intent(in)    :: val !< Value (> 0).
   integer(I4P)                    :: s   !< Slot.

   s = elisha_hash_slot(key, h%cap)
   do
      if (h%keys(s) == -1_I8P .or. h%keys(s) == key) exit
      s = mod(s + 1, h%cap)
   enddo
   h%keys(s) = key
   h%vals(s) = val
   endsubroutine put

   pure function get(h, key) result(val)
   !< The value of a key, 0 when absent.
   class(elisha_hash_table), intent(in) :: h    !< Table.
   integer(I8P),     intent(in) :: key  !< Key.
   integer(I4P)                 :: val  !< Value.
   integer(I4P)                 :: s, n !< Slot, probes.

   val = 0
   if (h%cap == 0) return
   s = elisha_hash_slot(key, h%cap)
   do n=1, h%cap
      if (h%keys(s) == -1_I8P) return
      if (h%keys(s) == key) then
         val = h%vals(s)
         return
      endif
      s = mod(s + 1, h%cap)
   enddo
   endfunction get

endmodule elisha_keys
