#include <assert.h>
#include <stdio.h>
#include "compiler.h"
#include "opt.h"
#include "requests.h"
#include "table.h"

#if 0
/** @param idx1 the 1D index of the brick (what got stored in the HT)
 * @param[out] bid output.  the corresponding 4D index from 'idx1'.
 * @param[in] bdims the number of bricks in each dimension */
static void
to4d(const unsigned idx1, unsigned bid[4], const unsigned bdims[4])
{
	bid[0] = idx1 % bdims[0];
	bid[1] = (idx1 / bdims[0]) % bdims[1];
	bid[2] = (idx1 / (bdims[0]*bdims[1])) % bdims[2];
	/* One really doesn't need the mod operation in the last elem.. */
	bid[3] = (idx1 / (bdims[0]*bdims[1]*bdims[2])) % bdims[3];
	if(serialize(bid, bdims) != idx1) {
		printf("dims: {%u,%u,%u,%u}\n",
		       bdims[0],bdims[1],bdims[2],bdims[3]);
		printf("div %u: %u\n", bdims[0], bid[1]);
		printf("%u x %u: %u\n", bdims[0], bdims[1], bdims[0]*bdims[1]);
		printf("%u serializes to [%u,%u,%u,%u] and back to %u!\n",
		       idx1, bid[0],bid[1],bid[2],bid[3],
		       serialize(bid,bdims));
	}
	assert(serialize(bid, bdims) == idx1);
}
#endif

/** removes a set of HT entries from the given bricktable.
 * @param entries the entries.  note these are 1D (collapsed) indices.
 * @param n_entries number of hash table entries
 * @param bricks the set of bricks, the brick table.
 * @param n_bricks number of bricks.  each brick is 4 elems!
 * @param bdims the number of bricks in each dimension
 * @returns the modified/new number of bricks in the table. */
size_t
remove_entries(const unsigned* entries, const size_t n_entries,
               unsigned* bricks, size_t n_bricks,
               const unsigned bdims[4])
{
	for(size_t i=0; i < n_entries; ++i) {
		if(entries[i] > 0) {
			n_bricks = remove_all(entries[i], bricks, n_bricks,
			                      bdims);
		}
	}
	return n_bricks;
}

/** @returns the number of nonzero entries in the array. */
PURE size_t
nonzeroes(const unsigned* ht, const size_t n_entries)
{
	size_t count = 0;
	for(size_t i=0; i < n_entries; ++i) {
		if(ht[i] != 0) { ++count; }
	}
	return count;
}

/** @returns the number of entries which are equal to 'value'. */
PURE static size_t
count(unsigned value, const unsigned* ht, const size_t n_entries)
{
	size_t n=0;
	for(size_t i=0; i < n_entries; ++i) {
		if(ht[i] == value) { ++n; }
	}
	return n;
}

/** @returns true if the array contains any value more than once. */
PURE bool
duplicates(const unsigned* ht, const size_t n_entries)
{
	for(size_t i=0; i < n_entries; ++i) {
		if(ht[i] > 0 && count(ht[i], ht, n_entries) > 1) {
			if(verbose()) {
				fprintf(stderr, "%u appears %zu times!\n",
				        ht[i], count(ht[i], ht, n_entries));
				return true;
			}
		}
	}
	return false;
}

/** to distinguish between empty elements and ones for a specific brick, the
 * serialization during insertion adds one.  This removes that +1 so that the
 * indices are again valid.
 * @param[inout] ht the hash table to fix
 * @param[in] n_entries number of valid indices in the hash table. */
void
subtract1(unsigned* ht, const size_t n_entries)
{
	for(size_t i=0; i < n_entries; ++i) {
		if(ht[i] > 0) {
			ht[i]--;
		}
	}
}
