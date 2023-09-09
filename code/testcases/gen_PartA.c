#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

int main(int argc, char* argv[]){
  int M, K, N;
  int i, j, k;
  printf("Input dimensions: <M> <K> <N>\n");
  scanf("%d %d %d", &M, &K, &N);
  if (M <= 0 || K <= 0 || N <= 0){
    printf("Dimensions should be positive!\n");
    exit(-1);
  }
  char input_buf[50], output_buf[50];
  sprintf(input_buf, "testcase_%d_%d_%d", M, K, N);
  sprintf(output_buf, "output_%d_%d_%d", M, K, N);

  srand(time(NULL));
  FILE* fw = fopen(input_buf, "w");
  fprintf(fw, "%d %d\n", M, K);

  int useRand = 1;
  #ifdef DEBUG
    useRand = 0;
  #endif

  if (!useRand){
    int count = 1;
    for (i=0; i<M; i++){
      for (j=0; j<K; j++){
        fprintf(fw, "%d ", count++);
      }
      fprintf(fw, "\n");
    }
    fprintf(fw, "\n%d %d\n", K, N);
    count = 1;
    for (i=0; i<K; i++){
      for (j=0; j<N; j++){
        fprintf(fw, "%d ", count++);
      }
      fprintf(fw, "\n");
    }
  } else {
    for (i=0; i<M; i++){
      for (j=0; j<K; j++){
        fprintf(fw, "%d ", rand() % 50);
      }
      fprintf(fw, "\n");
    }
    fprintf(fw, "\n%d %d\n", K, N);
    for (i=0; i<K; i++){
      for (j=0; j<N; j++){
        fprintf(fw, "%d ", rand() % 50);
      }
      fprintf(fw, "\n");
    }
  }
  fclose(fw);




  FILE *fr = fopen(input_buf, "r");
  int **A, **B, **C;
  fscanf(fr, "%d %d", &M, &K);
  A = (int**) malloc(sizeof(int*) * M);
  for (i = 0; i < M; i++){
    A[i] = (int*) malloc(sizeof(int) * K);
  }
  for (i = 0; i < M; i++){
    for (j = 0; j < K; j++){
      fscanf(fr, "%d", &A[i][j]);
    }
  }
  fscanf(fr, "%d %d", &K, &N);
  B = (int**) malloc(sizeof(int*) * K);
  for (i = 0; i < K; i++){
    B[i] = (int*) malloc(sizeof(int) * N);
  }
  for (i = 0; i < K; i++){
    for (j =0; j < N; j++){
      fscanf(fr, "%d", &B[i][j]);
    }
  }
  fclose(fr);
  
  C = (int**) malloc(sizeof(int*) * M);
  for (i = 0; i < M; i++){
    C[i] = (int*) malloc(sizeof(int) * N);
    memset(C[i], 0, sizeof(int) * N);
  }

  /** Sequential **/
  for (k = 0; k < K; k++){
    for (i = 0; i < M; i++){
      for (j = 0; j < N; j++){
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  fw = fopen(output_buf, "w");
  for (i = 0; i < M; i++){
    for (j = 0; j < N; j++){
      fprintf(fw, "%d ", C[i][j]);
    }
    fprintf(fw, "\n");
  }
  fclose(fw);
  return 0;
}