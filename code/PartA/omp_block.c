/**
 * @file omp_block.c
 * @brief blocked matrix multiplication with omp directives
 * update the intermediate multiplication values directly into shared memory.
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <string.h>
#include "../util/utils.h"

void blockMatMult(int ii, int bi, int kk, int bk, int jj, int bj);
void computeBlksz(int* bi, int* bj, int* bk);
void freeMem();

/** shared variables **/
int **A, **B, **C, *oracle;
int M, K, N, nthds;

int main(int argc, char *argv[]){
  if (argc != 3){
    printf("./omp_block_2 <num_threads> <input_filename>\n");
    exit(-1);
  }
  readInput(argv[2], &M, &K, &N, &A, &B, &C, &oracle);
  nthds = atoi(argv[1]);

  int i, j, k;
  int bi, bj, bk;
  computeBlksz(&bi, &bj, &bk);

  double start, end;

  /** static schedule **/
  for (i = 0; i < M; i++){
    memset(C[i], 0, sizeof(int)*N);
  }
  start = omp_get_wtime();
  #pragma omp parallel for num_threads(nthds) collapse (2) \
    shared(A, B, C, bi, bk, bj) private(i, j, k)
  for (i = 0; i < M; i+= bi){
    for (j = 0; j < N; j+= bj){
      // parallel region
      for (k = 0; k < K; k+= bk){
        blockMatMult(i,bi,k,bk,j,bj);
      }
    }
  }
  end = omp_get_wtime();
  printf("static: time = %.3fms\n", (end-start)*1000);
  checkCorrectness(M, N, C, oracle);
  
  /** dynamic schedule **/
  for (i = 0; i < M; i++){
    memset(C[i], 0, sizeof(int)*N);
  }
  start = omp_get_wtime();
  #pragma omp parallel for num_threads(nthds) collapse (2) \
    shared(A, B, C, bi, bk, bj) private(i, j, k) schedule(dynamic, 1)
  for (i = 0; i < M; i+= bi){
    for (j = 0; j < N; j+= bj){
      // parallel region
      for (k = 0; k < K; k+= bk){
        blockMatMult(i,bi,k,bk,j,bj);
      }
    }
  }
  end = omp_get_wtime();
  printf("dynamic: time = %.3fms\n", (end-start)*1000);
  checkCorrectness(M, N, C, oracle);

  /** static cyclic schedule **/
  for (i = 0; i < M; i++){
    memset(C[i], 0, sizeof(int)*N);
  }
  start = omp_get_wtime();
  #pragma omp parallel for num_threads(nthds) collapse (2) \
    shared(A, B, C, bi, bk, bj) private(i, j, k) schedule(static, 1)
  for (i = 0; i < M; i+= bi){
    for (j = 0; j < N; j+= bj){
      // parallel region
      for (k = 0; k < K; k+= bk){
        blockMatMult(i,bi,k,bk,j,bj);
      }
    }
  }
  end = omp_get_wtime();
  printf("static cylic: time = %.3fms\n", (end-start)*1000);
  checkCorrectness(M, N, C, oracle);

  freeMem();
  return 0;
}

/**
 * @brief 
 * submatrix multiplication that updates intermediate results directly into C matrix
 */
void blockMatMult(int ii, int bi, int kk, int bk, int jj, int bj){
  int i_lo = ii, j_lo = jj, k_lo = kk;
  int i_up = ii+bi <= M? ii+bi : M;
  int k_up = kk+bk <= K? kk+bk : K;
  int j_up = jj+bj <= N? jj+bj : N;
  
  // use k-i-j loop for optimal serial performance
  for (kk = k_lo; kk < k_up; kk++){
    for (ii = i_lo; ii < i_up; ii++){
      for (jj = j_lo; jj < j_up; jj++){
        // write through
        C[ii][jj] += A[ii][kk] * B[kk][jj];
      }
    }
  }
}

void computeBlksz(int* bi, int* bj, int* bk){
  int Pi, Pj; // thread grid size
  double ratio = M < N ? (double) N / (double) M : (double) M / (double) N;
  if (M < N){
    Pi = ceil(sqrt((double) nthds / ratio));
    Pj = ceil((double) Pi * ratio);
  }else {
    Pj = ceil(sqrt((double) nthds / ratio));
    Pi = ceil((double) Pj * ratio);
  }
  // block size
  *bi = *bj = *bk = ceil((double) M/(double) Pi);
}

void freeMem(){
  for (int i = 0; i < M; i++){
    free(A[i]); free(C[i]);
  }
  for (int i = 0; i < K; i++){
    free(B[i]);
  }
  free(A); free(B); free(C); free(oracle);
}