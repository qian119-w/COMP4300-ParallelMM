#include <stdio.h>
#include <stdlib.h>

/* part A */
void readInput(char* input_filename, int* Mp, int* Kp, int* Np, int*** A, int*** B, int*** C, int** oracle);
void mpi_readInputC(char* input_filename, int* MKNp, int*** A, int*** B, int*** C, int** oracle);
void checkCorrectness(int M, int N, int** C, int* oracle);
void mallocCont(int*** array, int n, int m);

/* part B */
void readSorting(char *filename, int* n, int** a, int** oracle);
void checkSorting(int n, int* a, int* oracle);