#ifndef TJF_HT_OPT_H
#define TJF_HT_OPT_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef WIN32
#include <stdbool.h>
#endif

void argparse(int argc, char* argv[]);

size_t htN();
size_t bricksX();
size_t bricksY();
size_t bricksZ();
size_t LODs();
const char* requestfile();
size_t blockingfactor();
bool naive();
bool verbose();

#ifdef __cplusplus
}
#endif
#endif /* TJF_HT_OPT_H */
