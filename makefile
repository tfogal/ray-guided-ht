GLIB_CF:=$(shell pkg-config --cflags glib-2.0)
GLIB_LD:=$(shell pkg-config --libs glib-2.0)
CFLAGS:=-ggdb -Wall -Wextra $(GLIB_CF) -std=c99
CFLAGS:=-ggdb -Wall $(GLIB_CF) -std=c99 -O3
CXXFLAGS=-ggdb -Wall -Wextra $(GLIB_CF) -std=c++11
CXXFLAGS=-ggdb -Wall -Wextra $(GLIB_CF) -std=c++11 -O3
NVFLAGS:=-g -G $(GLIB_CF) -arch=sm_30
NVFLAGS:=$(GLIB_CF) -arch=sm_30 -O3
LIBS:=$(GLIB_LD)
NVCC:=nvcc
OBJ=ht.o opt.o requests.o table.o ba-file.o SysTools.o

all: $(OBJ) Hash

Hash: ba-file.o ht.o opt.o requests.o SysTools.o table.o
	$(NVCC) $(NVFLAGS) $^ -o $@ $(LIBS)

%.o: %.cu
	$(NVCC) $(NVFLAGS) -c $^ -o $@

clean:
	rm -f Hash $(OBJ)
