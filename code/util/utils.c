#include "utils.h"

/**
 * @brief 
 * read from file
 * for OpenMP
 */
void readInput(char* input_filename, int* Mp, int* Kp, int* Np, int*** A,
int*** B, int*** C, int** oracle){
  int i,j;
  FILE* fr = fopen(input_filename, "r");
  fscanf(fr, "%d %d", Mp, Kp);
  *A = (int**) malloc(sizeof(int*) * (*Mp));
  for (i = 0; i < (*Mp); i++){
    (*A)[i] = (int*) malloc(sizeof(int) * (*Kp));
  }
  for (i = 0; i < (*Mp); i++){
    for (j = 0; j < (*Kp); j++){
      fscanf(fr, "%d", &(*A)[i][j]);
    }
  }
  fscanf(fr, "%d %d", Kp, Np);
  *B = (int**) malloc(sizeof(int*) * (*Kp));
  for (i = 0; i < (*Kp); i++){
    (*B)[i] = (int*) malloc(sizeof(int) * (*Np));
  }
  for (i = 0; i < (*Kp); i++){
    for (j =0; j < (*Np); j++){
      fscanf(fr, "%d", &(*B)[i][j]);
    }
  }
  *C = (int**) malloc(sizeof(int*) * (*Mp));
  for (i = 0; i < (*Mp); i++){
    (*C)[i] = (int*) malloc(sizeof(int) * (*Np));
  }
  fclose(fr);

  char file[50];
  sprintf(file, "../testcases/output_%d_%d_%d", *Mp, *Kp, *Np);
  *oracle = malloc(sizeof(int)* (*Mp) * (*Np));
  fr = fopen(file, "r");
  for (int i = 0; i < (*Mp) * (*Np); i++){
    fscanf(fr, "%d", &(*oracle)[i]);
  }
  fclose(fr);
}


void mallocCont(int*** array, int n, int m){
  int* p = (int*) malloc(sizeof(int)*n*m);
  *array = (int**) malloc(sizeof(int*)*n);
  for (int i = 0; i<n; i++){
    (*array)[i] = &p[i*m];
  }
}

/**
 * @brief 
 * Use 1d array to simulate 2d matrices 
 * for MPI and Pthread
 */
void mpi_readInputC(char* input_filename, int* MKNp, int*** A,
int*** B, int*** C, int** oracle){
  int i,j;
  FILE* fr = fopen(input_filename, "r");
  fscanf(fr, "%d %d", &MKNp[0], &MKNp[1]);
  
  mallocCont(A, MKNp[0], MKNp[1]);
  for (i = 0; i < MKNp[0]; i++){
    for (j = 0; j < MKNp[1]; j++){
      fscanf(fr, "%d", &(*A)[i][j]);
    }
  }
  fscanf(fr, "%d %d", &MKNp[1], &MKNp[2]);
  mallocCont(B, MKNp[1], MKNp[2]);
  for (i = 0; i < MKNp[1]; i++){
    for (j =0; j < MKNp[2]; j++){
      fscanf(fr, "%d", &(*B)[i][j]);
    }
  }
  mallocCont(C, MKNp[0], MKNp[2]);
  fclose(fr);

  char file[50];
  sprintf(file, "../testcases/output_%d_%d_%d", MKNp[0], MKNp[1], MKNp[2]);
  *oracle = malloc(sizeof(int)* MKNp[0] * MKNp[2]);
  fr = fopen(file, "r");
  for (int i = 0; i < MKNp[0] * MKNp[2]; i++){
    fscanf(fr, "%d", &(*oracle)[i]);
  }
  fclose(fr);
}

/**
 * @brief 
 * check matmul computation correctness
 */
void checkCorrectness(int M, int N, int** C, int* oracle){
  int i, j;
  int bool = 0;
  for (i = 0; i < M; i++){
    for (j = 0; j < N; j++){
      if (C[i][j] != oracle[i*N+j]){
        bool = 1;
        goto end;
      }
    }
  }
  end:
  if (bool) printf("Mat Mul incorrect\n");
  else printf("Mat Mul correct\n");
}

/**
 * @brief 
 * check sorting correctness
 */
void readSorting(char *filename, int* n, int** a, int** oracle){
  int i;
  FILE* fr = fopen(filename, "r");
  fscanf(fr, "%d", n);
  *a = malloc(sizeof(int)* *n);
  *oracle = malloc(sizeof(int)* *n);
  for (i =0; i< *n; i++){
    fscanf(fr, "%d", &(*a)[i]);
  }
  fclose(fr);

  char output_buf[50];
  sprintf(output_buf, "../testcases/output_sorting_%d", *n);
  fr = fopen(output_buf, "r");
  for (i = 0; i< *n; i++){
    fscanf(fr, "%d", &(*oracle)[i]);
  }
  fclose(fr);
}

void checkSorting(int n, int* a, int* oracle){
  int i;
  int incorrect = 0;
  for (i = 0; i < n; i++){
    if (a[i] != oracle[i]){
      printf("a[%d]=%d vs %d\n", i, a[i], oracle[i]);
      incorrect = 1;
    }
  }
  if (incorrect) printf("Sorting incorrect\n");
  else printf("Sorting correct\n");
}