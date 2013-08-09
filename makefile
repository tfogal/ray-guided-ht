GLIB_CF:=$(shell pkg-config --cflags glib-2.0)
GLIB_LD:=$(shell pkg-config --libs glib-2.0)
CFLAGS:=-ggdb -Wall $(GLIB_CF)
CFLAGS:=-ggdb -Wall $(GLIB_CF) -O3
NVFLAGS:=-g -G $(GLIB_CF) -arch=sm_30
NVFLAGS:=-O3 $(GLIB_CF) -arch=sm_30
LIBS:=$(GLIB_LD)
NVCC:=nvcc
OBJ=ht.o opt.o

all: $(OBJ) Hash

Hash: ht.o opt.o
	$(NVCC) $(NVFLAGS) $^ -o $@ $(LIBS)

%.o: %.cu
	$(NVCC) $(NVFLAGS) -c $^ -o $@

clean:
	rm -f Hash $(OBJ)
