#ifndef TJF_HT_OPT_H
#define TJF_HT_OPT_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

void argparse(int argc, char* argv[]);

size_t htN();
size_t bricksX();
size_t bricksY();
size_t bricksZ();
size_t LODs();
const char* requestfile();

#ifdef __cplusplus
}
#endif
#endif /* TJF_HT_OPT_H */
