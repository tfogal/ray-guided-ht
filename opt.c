#include <stdlib.h>
#include <glib.h>
#include "opt.h"

static gboolean debug;
static gint ht_size;
static gint bricks[3]; /* per-dimension. */
static gint n_lod;
static gchar* requestf; /* where to read requests from */
static gint blockfactor;
static bool ht_naive;
static bool is_verbose;
static GOptionEntry options[] = {
	{"debug", 'd', 0, G_OPTION_ARG_NONE, &debug, "Debug mode", NULL},
	{"hashsz", 'h', 0, G_OPTION_ARG_INT, &ht_size, "HT size in X", NULL},
	{"bx", 'X', 0, G_OPTION_ARG_INT, &bricks[0], "N bricks in X", NULL},
	{"by", 'Y', 0, G_OPTION_ARG_INT, &bricks[1], "N bricks in Y", NULL},
	{"bz", 'Z', 0, G_OPTION_ARG_INT, &bricks[2], "N bricks in Z", NULL},
	{"lods", 'l', 0, G_OPTION_ARG_INT, &n_lod, "number of LODs", NULL},
	{"requests", 'r', 0, G_OPTION_ARG_FILENAME, &requestf,
	 "requests to simulate, file", NULL},
	{"blockfactor", 'b', 0, G_OPTION_ARG_INT, &blockfactor,
	 "CUDA kernel blocking factor; divide bricks by this", NULL},
	{"naive", 'n', false, G_OPTION_ARG_NONE, &ht_naive,
	 "use naive hash table algorithm (clone of IV3D's GLSL)", NULL},
	{"verbose", 'v', false, G_OPTION_ARG_NONE, &is_verbose,
	 "enable chatty/annoying output.", NULL},
	{ NULL, 0, 0, G_OPTION_ARG_NONE, NULL, NULL, NULL }
};

void
argparse(int argc, char* argv[])
{
	GError* error = NULL;
	GOptionContext* ctx = g_option_context_new("- ht testing");
	g_option_context_add_main_entries(ctx, options, 0);
	g_option_context_parse(ctx, &argc, &argv, &error);

	g_option_context_free(ctx);

	if(ht_size <= 0) {
		g_critical("HT must have a positive size.");
		exit(EXIT_FAILURE);
	}
	if(bricks[0] <= 0) {
		g_critical("number of X bricks must be positive.");
		exit(EXIT_FAILURE);
	}
	if(bricks[1] <= 0) {
		g_critical("number of Y bricks must be positive.");
		exit(EXIT_FAILURE);
	}
	if(bricks[1] <= 0) {
		g_critical("number of Z bricks must be positive.");
		exit(EXIT_FAILURE);
	}
	const size_t nbricks = (size_t)bricks[0]*bricks[1]*bricks[2]*n_lod;
	if(nbricks < ht_size) {
		g_warning("more hash table entries (%d) than bricks (%zu)!",
		          ht_size, nbricks);
	}
}

size_t htN() { return (size_t)ht_size; }
size_t bricksX() { return (size_t)bricks[0]; }
size_t bricksY() { return (size_t)bricks[1]; }
size_t bricksZ() { return (size_t)bricks[2]; }
size_t LODs() { return (size_t)n_lod; }
const char* requestfile() { return (const char*)requestf; }
size_t blockingfactor() { return (size_t)blockfactor; }
bool naive() { return ht_naive; }
bool verbose() { return is_verbose; }
