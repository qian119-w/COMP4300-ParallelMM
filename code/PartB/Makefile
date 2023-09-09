CC = gcc
MPICC = mpicc
CUDACC = nvcc
OMP_FLAG = -fopenmp
PTH_FLAG = -pthread
CFLAGS	= -g -Wall -O3
LDFLAGS	= -lm
UDIR = ../util

_DEPS = utils.h utils.c
DEPS = $(patsubst %, $(UDIR)/%, $(_DEPS))

all: pth_sort pth_sort_2 omp_sort omp_sort_2 mpi_sort mpi_sort_2 cuda_sort cuda_sort_2

pth_sort: pth_sort.c $(DEPS)
	${CC} ${CFLAGS} ${PTH_FLAG} -o $@ $^ ${LDFLAGS}

pth_sort_2: pth_sort_2.c $(DEPS)
	${CC} ${CFLAGS} ${PTH_FLAG} -o $@ $^ ${LDFLAGS}

omp_sort: omp_sort.c $(DEPS)
	${CC} ${CFLAGS} ${OMP_FLAG} -o $@ $^ ${LDFLAGS}

omp_sort_2: omp_sort_2.c $(DEPS)
	${CC} ${CFLAGS} ${OMP_FLAG} -o $@ $^ ${LDFLAGS}

mpi_sort: mpi_sort.c $(DEPS)
	${MPICC} ${CFLAGS} -o $@ $^ ${LDFLAGS}

mpi_sort_2: mpi_sort_2.c $(DEPS)
	${MPICC} ${CFLAGS} -o $@ $^ ${LDFLAGS}

cuda_sort: cuda_sort.cu
	${CUDACC} -o $@ $^ ${LDFLAGS}

cuda_sort_2: cuda_sort_2.cu
	${CUDACC} -o $@ $^ ${LDFLAGS}