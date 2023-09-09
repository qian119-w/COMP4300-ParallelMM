/**
 * @file omp_block_2.c
 * @brief 
 * using private thread memory to store intermediate multiplication results.
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <string.h>
#include "../util/utils.h"

void blockMatMult(int* temp, int ii, int bi, int kk, int bk, int jj, int bj);
void computeBlksz(int* bi, int* bj, int* bk);
void freeMem();

/** shared variables **/
int **A, **B, **C, *oracle;
int M, K, N, nthds;
int max_bz = 500;

int main(int argc, char *argv[]){
  if (argc != 3){
    printf("./omp_block <num_threads> <input_filename>\n");
    exit(-1);
  }
  // read from input file
  readInput(argv[2], &M, &K, &N, &A, &B, &C, &oracle);
  nthds = atoi(argv[1]);

  int i, j, k;
  int bi, bj, bk;
  computeBlksz(&bi, &bj, &bk);

  for (i = 0; i < M; i++){
    memset(C[i], 0, sizeof(int)*N);
  }
  int real_nthds;
  double start, end;
  start = omp_get_wtime();
  #pragma omp parallel num_threads(nthds) \
    shared(A, B, C, bi, bk, bj) private(i, j, k)
  {
    if (omp_get_thread_num() == 0)
      real_nthds = omp_get_num_threads();
    int* temp = malloc(sizeof(int)*bi*bj);
    #pragma omp for collapse (2) schedule(static, 1)
    for (i = 0; i < M; i+= bi){
      for (j = 0; j < N; j+= bj){
        // parallel region
        memset(temp, 0, sizeof(int)*bi*bj);
        for (k = 0; k < K; k+= bk){
          blockMatMult(temp,i,bi,k,bk,j,bj);
        }
      }
    }
    free(temp);
  }
  end = omp_get_wtime();
  printf("static: nthds = %d time = %.3fms\n", real_nthds, (end-start)*1000);
 
  checkCorrectness(M, N, C, oracle);

  for (i = 0; i < M; i++){
    memset(C[i], 0, sizeof(int)*N);
  }
  start = omp_get_wtime();
  #pragma omp parallel num_threads(nthds) \
    shared(A, B, C, bi, bk, bj) private(i, j, k)
  {
    if (omp_get_thread_num() == 0)
      real_nthds = omp_get_num_threads();
    int* temp = malloc(sizeof(int)*bi*bj);
    #pragma omp for collapse (2) schedule(dynamic, 1)
    for (i = 0; i < M; i+= bi){
      for (j = 0; j < N; j+= bj){
        memset(temp, 0, sizeof(int)*bi*bj);
        for (k = 0; k < K; k+= bk){
          blockMatMult(temp,i,bi,k,bk,j,bj);
        }
      }
    }
    free(temp);
  }
  end = omp_get_wtime();
  printf("dynamic: nthds = %d time = %.3fms\n", real_nthds, (end-start)*1000);
  checkCorrectness(M, N, C, oracle);

  freeMem();
  return 0;
}

/**
 * @brief 
 * submatrix multiplication that "cache" the intermediate results in private memory
 */
void blockMatMult(int* temp, int ii, int bi, int kk, int bk, int jj, int bj){
  int i_lo = ii, j_lo = jj, k_lo = kk;
  int i_up = ii+bi <= M? ii+bi : M;
  int k_up = kk+bk <= K? kk+bk : K;
  int j_up = jj+bj <= N? jj+bj : N;
  int count;

  // use k-i-j loop for optimal serial performance
  for (kk = k_lo; kk < k_up; kk++){
    count = 0;
    for (ii = i_lo; ii < i_up; ii++){
      for (jj = j_lo; jj < j_up; jj++){
        // cache in private memory
        temp[count++] += A[ii][kk] * B[kk][jj];
      }
    }
  }
  // write back
  count = 0;
  for (ii = i_lo; ii < i_up; ii++){
    for (jj= j_lo; jj < j_up; jj++){
      C[ii][jj] = temp[count++];
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

  // block size bi * bj
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