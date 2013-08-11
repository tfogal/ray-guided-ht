#ifndef TJF_HT_TABLE_H
#define TJF_HT_TABLE_H
/* Helper functions for dealing with the hash table */
#include <stdbool.h>
#include <stdlib.h>
#include "compiler.h"

#ifdef __cplusplus
extern "C" {
#endif

/** removes a set of HT entries from the given bricktable.
 * @param entries the entries.  note these are 1D (collapsed) indices.
 * @param n_entries number of hash table entries
 * @param bricks the set of bricks, the brick table.
 * @param n_bricks number of bricks.  each brick is 4 elems!
 * @param bdims the number of bricks in each dimension
 * @returns the modified/new number of bricks in the table. */
extern size_t
remove_entries(const unsigned* entries, const size_t n_entries,
               unsigned* bricks, size_t n_bricks,
               const unsigned bdims[4]);

PURE extern size_t
nonzeroes(const unsigned* ht, const size_t n_entries);

PURE extern bool
duplicates(const unsigned* ht, const size_t n_entries);

/** to distinguish between empty elements and ones for a specific brick, the
 * serialization during insertion adds one.  This removes that +1 so that the
 * indices are again valid.
 * @param[inout] ht the hash table to fix
 * @param[in] n_entries number of valid indices in the hash table. */
extern void
subtract1(unsigned* ht, const size_t n_entries);

#ifdef __cplusplus
}
#endif
#endif
