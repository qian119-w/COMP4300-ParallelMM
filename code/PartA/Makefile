CC = gcc
MPICC = mpicc
OMP_FLAG = -fopenmp
PTH_FLAG = -pthread
CFLAGS	= -g -Wall -O3
LDFLAGS	= -lm
UDIR = ../util

_DEPS = utils.h utils.c
DEPS = $(patsubst %, $(UDIR)/%, $(_DEPS))

all: omp_block omp_block_2 mpi_summa mpi_summa_2 \
 s2 mpi_cannon mpi_cannon_2 srr pth_summa pth_summa_2 pth_cannon pth_cannon_2

omp_block: omp_block.c $(DEPS)
	${CC} ${CFLAGS} ${OMP_FLAG} -o $@ $^ ${LDFLAGS}

omp_block_2: omp_block_2.c $(DEPS)
	${CC} ${CFLAGS} ${OMP_FLAG} -o $@ $^ ${LDFLAGS}

pth_summa: pth_summa.c $(DEPS)
	${CC} ${CFLAGS} ${PTH_FLAG} -o $@ $^ ${LDFLAGS}

pth_summa_2: pth_summa_2.c $(DEPS)
	${CC} ${CFLAGS} ${PTH_FLAG} -o $@ $^ ${LDFLAGS}

pth_cannon: pth_cannon.c $(DEPS)
	${CC} ${CFLAGS} ${PTH_FLAG} -o $@ $^ ${LDFLAGS}

pth_cannon_2: pth_cannon_2.c $(DEPS)
	${CC} ${CFLAGS} ${PTH_FLAG} -o $@ $^ ${LDFLAGS}

mpi_summa: mpi_summa.c $(DEPS)
	${MPICC} ${CFLAGS} -o $@ $^ ${LDFLAGS}

mpi_summa_2: mpi_summa_2.c $(DEPS)
	${MPICC} ${CFLAGS} -o $@ $^ ${LDFLAGS}

mpi_cannon: mpi_cannon.c $(DEPS)
	${MPICC} ${CFLAGS} -o $@ $^ ${LDFLAGS}

mpi_cannon_2: mpi_cannon_2.c $(DEPS)
	${MPICC} ${CFLAGS} -o $@ $^ ${LDFLAGS}

s2: s2.c
	${MPICC} ${CFLAGS} -o $@ $^ ${LDFLAGS}

srr: srr.c
	${MPICC} ${CFLAGS} -o $@ $^ ${LDFLAGS}

clean:
	/bin/rm -f *.o *~ omp_block omp_block_2 baseline block_sz mpi_summa mpi_summa_2 \
	mpi_cannon mpi_cannon_2 pth_summa pth_summa_2 pth_cannon pth_cannon_2