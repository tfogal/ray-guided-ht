#ifndef TJF_HT_COMPILER_H
#define TJF_HT_COMPILER_H

#ifdef __GNUC__
#	define PURE __attribute__((pure))
#	define CONST __attribute__((const))
#else
#	define PURE /* no pure function support */
#	define CONST /* no const function support */
#endif

#endif
