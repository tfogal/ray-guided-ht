#ifndef TJF_HT_REQUESTS_H
#define TJF_HT_REQUESTS_H
#include <inttypes.h>
#include <stdbool.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/** reads requests from the given filename.
 * @returns the array of requests, or NULL on error.
 * @param[out] nreqs the number of requests in the array; note the array
 *             then has 4*requests elements, since each requests is 4
 *             entries. */
extern uint32_t* requests_from(const char* filename, size_t* nreqs);

/* are the given requests valid?  they need to fall within brick indices.
 * @param requests the requests the examine
 * @param nreq number of requests; 'requests' is 4*nreq elems long.
 * @param bdims the brick dimensions.
 * @param[out] erridx if nonnull, the request which was in error. */
extern bool requests_verify(const uint32_t* requests, const size_t nreq,
                            const unsigned bdims[4], size_t* erridx);

size_t
idx_of(const uint32_t* requests, const size_t nreq, const unsigned val,
       const unsigned bdims[4]);

/** removes all entries from the request table which serialize to the given
 * value.
 * @param[in] serized the serialized value to be removed
 * @param[inout] bricks the table of bricks to modify
 * @param[in] n_bricks the number of bricks in the given table.
 * @return the new/updated number of bricks in the table */
extern size_t
remove_all(unsigned serized, unsigned* bricks, size_t n_bricks,
           const unsigned bdims[4]);

extern void
write_requests(const char* file, const unsigned* bricks, size_t n_bricks);

#ifdef __cplusplus
}
#endif
#endif
