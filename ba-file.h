#ifndef HT_BRICK_ACCESS_FILE_H
#define HT_BRICK_ACCESS_FILE_H

#include <cstdlib>

/** loads requests from a '.ba' file, in the format the HT wants. */
extern unsigned* requests_ba(const char* filename, size_t* n);

/** loads nbricks etc. from a .ba file. */
extern void brickdims_ba(const char* filename, unsigned brickdims[4]);

extern bool is_ba(const char* filename);

#endif // BRICK_ACCESS_FILE_H
