#ifndef HT_BRICK_ACCESS_FILE_H
#define HT_BRICK_ACCESS_FILE_H

#include <cstdlib>

/** loads requests from a '.ba' file, in the format the HT wants. */
extern unsigned* requests_ba(const char* filename, size_t* n);

#endif // BRICK_ACCESS_FILE_H
