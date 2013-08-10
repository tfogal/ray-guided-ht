#ifndef TJF_HT_COMPILER_H
#define TJF_HT_COMPILER_H

#ifdef __GNUC__
#	define CONST __attribute__((const))
#	define MALLOC __attribute__((malloc))
#	define PURE __attribute__((pure))
#else
#	define CONST /* no const function support */
#	define MALLOC /* no malloc function support */
#	define PURE /* no pure function support */
#endif

#endif
