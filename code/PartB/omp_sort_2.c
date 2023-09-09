/**
 * @file omp_sort.c
 * @brief 
 * parallel sample sort with OpenMP
 * 
 * broadcast in shared memory
 * final splitter selection in private memory by each thread
 * 
 * block assignment to simulate random bucket assignment for splitter selection
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <string.h>
#include "../util/utils.h"

int cmpfunc(const void *a, const void *b) {
  return (*(int*)a - *(int*)b);
}
#define MAX __INT_MAX__


int main(int argc, char* argv[]){
  int n, nthds;
  int* a, *oracle, *output;
  if (argc != 3){
    printf("./omp_sort <num_threads> <input_filename>\n");
    exit(-1);
  }
  /* read from file */
  readSorting(argv[2], &n, &a, &oracle);

  nthds = atoi(argv[1]);
  if (n < (nthds*nthds)) {
    printf("sample sort: n > square(nthds)\n");
    exit(-1);
  }
  output = malloc(sizeof(int)*n);

  /* shared */
  int* global_splitters = malloc(sizeof(int) * (nthds-1) * nthds);
  int* bucket_counts = malloc(sizeof(int) * nthds);

  /* performance measurement */
  double start, end;
  start = omp_get_wtime();
  
  int i, count, d, r, par;
  #pragma omp parallel num_threads(nthds) \
    private(i, count, d, r, par) shared(global_splitters, bucket_counts)
  {
    int my_rank, comm_sz;
    comm_sz = omp_get_num_threads();
    my_rank = omp_get_thread_num();
    int length = n / comm_sz;
    int* subarray = malloc(sizeof(int)*length);
    count = 0;
    /* block assignment for splitter selection (pseudo random) */
    #pragma omp for
    for (i = 0; i < n; i++){
      if (count == length){
        length *=2;
        subarray = realloc(subarray, sizeof(int)*length);
      }
      subarray[count++] = a[i];
    }
    /* local splitter selection */
    qsort(subarray, count, sizeof(int), cmpfunc);
    int* local_splitters = malloc(sizeof(int) * (comm_sz-1));
    d = count / comm_sz;
    r = count % comm_sz;
    for (i = 1, par = 0; i < comm_sz; i++){
      par += (i < r ? d+1 : d); 
      local_splitters[i-1] = subarray[par];
    }
    free(subarray);

    /* broadcast local splitters in shared memory */
    int start_pos = (comm_sz-1)*my_rank;
    for (i = 0; i < comm_sz-1; i++){
      global_splitters[start_pos+i] = local_splitters[i];
    }
    #pragma omp barrier
    
    /* global splitter sort in shared memory by Thread 0*/
    if (my_rank == 0){
      qsort(global_splitters, (comm_sz-1) * comm_sz, sizeof(int), cmpfunc);
    }
    #pragma omp barrier

    /* final splitter selection in private memory by all threads */
    int* final_splitters = malloc(sizeof(int) * (nthds-1));
    d = comm_sz / 2;
    for (i = 0; i < comm_sz-1; i++){
      final_splitters[i] = global_splitters[d+i*comm_sz];
    }

    /* bucket sort */
    int bucket_low, bucket_high;
    bucket_low = my_rank == 0? -1 : final_splitters[my_rank-1];
    bucket_high = my_rank == comm_sz-1 ? MAX : final_splitters[my_rank];
    int my_bucket_size = n / comm_sz;
    int* my_bucket = malloc(sizeof(int) * my_bucket_size);
    for (i = 0, count = 0; i< n; i++){
      if (a[i] > bucket_low && a[i] <= bucket_high){
        if (count == my_bucket_size){
          my_bucket_size *=2;
          my_bucket = realloc(my_bucket, sizeof(int)*my_bucket_size);
        }
        my_bucket[count++] = a[i];
      }
    }
    /* broadcast bucket count in shared memory */
    bucket_counts[my_rank] = count;
    #pragma omp barrier

    /* write result to correct position */
    for (i = 0, start_pos = 0; i< my_rank; i++){
      start_pos += bucket_counts[i];
    }
    qsort(my_bucket, count, sizeof(int), cmpfunc);
    for (i = 0; i < count; i++){
      output[start_pos+i] = my_bucket[i];
    }

    free(final_splitters);
    free(local_splitters);
    free(my_bucket);
  }

  /* correctness checking & timing output*/
  end = omp_get_wtime();
  checkSorting(n, output, oracle);
  printf("computed in %.8fs\n", end-start);
  free(a); free(oracle); free(output);
  free(global_splitters);
  free(bucket_counts);
  return 0;
}