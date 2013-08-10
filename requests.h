#ifndef TJF_HT_REQUESTS_H
#define TJF_HT_REQUESTS_H
#include <stdint.h>
#include <stdlib.h>

#ifndef WIN32
#include <stdbool.h>
#endif

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

#ifdef __cplusplus
}
#endif
#endif
